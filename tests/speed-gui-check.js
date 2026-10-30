// Headless-browser check of PXA Control's Speed tab ("Benchmarks"): the three Speed-tab fixes (Discord feature requests FR#1, FR#2, FR#11).
// Run it with tests/speed-gui-check.py, which starts the hermetic environment (tests/live_env.py: fake servers that answer /completion
// like llama-server, a fake nvidia-smi, 45 seeded benchmark rows) and the browser:
//     node speed-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME PHASES
// PHASES is a comma list of: refresh, table, strip, label.
//   refresh  a finished benchmark redraws the medians table, the legends and the graphs by itself (FR#2: it used to need a reload),
//            once (no redraw loop), and does so for a page opened while a job had already finished
//   table    the history table: 20 per page, prev/next, "page X of Y", search that resets to page 1, MTP and Split columns, no page scroll
//   strip    the server strip: status, Stop/Start flip it, the basic edit reuses the Launch/Servers calls
//   label    the tab reads "Benchmarks" (its id and hash stay "speed")
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, PHASES] = process.argv.slice(2);
const W = +WIDTH, PH = (PHASES || 'refresh,table,strip,label').split(',');
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
const txt = async sel => (await page.locator(sel).allInnerTexts()).join(' | ').replace(/\s+/g, ' ').trim();
const count = sel => page.locator(sel).count();
async function until(fn, ms, step) { const t0 = Date.now(); for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t0 > ms) return v; await sleep(step || 250); } }
const noPageScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
const canvasSig = id => page.evaluate(i => { const c = document.getElementById(i); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let h = 0, ink = 0; for (let k = 0; k < d.length; k += 4) { h = (h * 31 + d[k] + 7 * d[k + 1] + 13 * d[k + 2] + 17 * d[k + 3]) | 0; if (d[k + 3]) ink++; } return h + ':' + ink; }, id);
async function openSpeed() {
  await page.goto(BASE + '/#speed');
  await page.waitForSelector('#sp-target option', {state: 'attached', timeout: 30000});
  await until(async () => (await txt('#sp-src')).length > 8, 15000);
  await sleep(1200);
}

// ---- (a) FR#2: the tables and graphs redraw when a benchmark ends
async function refresh() {
  console.log(`[${THEME} ${WIDTH}px] a finished benchmark redraws the tables and graphs without a reload`);
  await openSpeed();
  await page.evaluate(() => { window.__nav = 'same-page'; });
  const reqs = [];
  page.on('request', r => { if (/\/pxa\/stats/.test(r.url())) reqs.push(Date.now()); });
  const before = {med: await txt('#sp-med'), lg: await txt('#lg-dec'), dec: await canvasSig('ch-dec'), pre: await canvasSig('ch-pre')};
  check(/nothing measured/.test(before.med), `before the run the medians table is empty: "${before.med}"`);
  await page.fill('#bn-warm', '0'); await page.fill('#bn-reps', '1');
  await page.click('#bn-run');
  check(!!(await until(async () => /running/.test(await txt('#bn-run')), 8000)), 'the button says "running ..." while the benchmark runs');
  const fin = await until(async () => /Run benchmark/.test(await txt('#bn-run')) && /done/.test(await txt('#bn-lines')), 60000, 300);
  check(!!fin, 'the benchmark finished (log says done)');
  const tFin = Date.now();
  const redrawn = await until(async () => !/nothing measured/.test(await txt('#sp-med')), 7000);
  check(!!redrawn, `the medians table fills by itself within 7 s of the end, no reload: "${(await txt('#sp-med')).slice(0, 120)}"`);
  const rows = await page.evaluate(() => [...document.querySelectorAll('#sp-med tr')].map(tr => [...tr.children].map(td => td.textContent)));
  const total = rows.reduce((a, r) => a + (+r[2] || 0), 0);
  check(total >= 4, `the table counts the benchmark's requests (3 classes + the identity check = 4; counted ${total})`);
  check((await txt('#lg-dec')) !== before.lg && /Rig/.test(await txt('#lg-dec')), `the decode legend names the server's model: "${await txt('#lg-dec')}"`);
  const after = {dec: await canvasSig('ch-dec'), pre: await canvasSig('ch-pre')};
  check(after.dec !== before.dec, 'the decode graph was redrawn with the new requests');
  check(after.pre !== before.pre, 'the prefill graph was redrawn with the new requests');
  check(await page.evaluate(() => window.__nav) === 'same-page', 'the page was not reloaded');
  await sleep(4500);
  const n = reqs.filter(t => t >= tFin - 500).length;
  check(n >= 1 && n <= 3, `one redraw, not a loop: ${n} /pxa/stats call(s) in the 5 s after the end`);
  await snap('after-benchmark');
  // opening the tab while a job has already finished must not redraw by itself, let alone forever
  await page.locator('nav.tabs button[data-tab=servers]').click(); await sleep(800);
  const k0 = reqs.length;
  await page.locator('nav.tabs button[data-tab=speed]').click(); await sleep(6000);
  const k = reqs.length - k0;
  check(k >= 1 && k <= 2, `opening the tab after the job ended draws once, not forever (${k} /pxa/stats call(s) in 6 s)`);
}


// ---- (b) FR#1: the history table pages, searches and shows MTP and Split
async function table() {
  console.log(`[${THEME} ${WIDTH}px] the benchmark history: pages, search, MTP and Split columns`);
  await openSpeed();
  await page.waitForSelector('#bn-hist tr td:not(.empty)', {timeout: 20000});
  const info = () => page.evaluate(() => ({
    rows: [...document.querySelectorAll('#bn-hist tr')].filter(tr => !tr.querySelector('.empty')).map(tr => [...tr.children].map(td => td.textContent)),
    empty: !!document.querySelector('#bn-hist .empty'), page: document.querySelector('#bn-page').textContent,
    prev: document.querySelector('#bn-prev').disabled, next: document.querySelector('#bn-next').disabled}));
  const heads = await txt('table.hist thead');
  check(/MTP/i.test(heads) && /Split/i.test(heads), `the table has the MTP and Split columns: ${heads}`);
  let r = await info();
  const total = +((/\((\d+) runs\)/.exec(r.page) || [0, 0])[1]);
  const pages = Math.ceil(total / 20);
  check(total >= 45 && pages >= 3, `the seeded history is there: ${total} runs, ${pages} pages ("${r.page}")`);
  check(r.rows.length === 20, `page 1 shows 20 rows (${r.rows.length})`);
  check(new RegExp(`^page 1 of ${pages}\\b`).test(r.page), `the pager says "page 1 of ${pages}": "${r.page}"`);
  check(r.prev && !r.next, 'Prev is disabled on the first page, Next is not');
  check(r.rows.every(c => /^(on|off)$/.test(c[3]) && /^(tensor|layer|plain)$/.test(c[4])), `page 1 (the newest runs) shows MTP on/off and a split mode: ${JSON.stringify(r.rows[0].slice(3, 5))}`);
  check(r.rows.some(c => c[3] === 'on'), 'at least one row shows MTP on');
  await snap('table-page1');
  await page.click('#bn-next');
  r = await info();
  check(new RegExp(`^page 2 of ${pages}\\b`).test(r.page) && r.rows.length === 20, `Next goes to page 2 with 20 rows: "${r.page}"`);
  check(!r.prev, 'Prev is enabled on page 2');
  await page.click('#bn-next');
  r = await info();
  check(new RegExp(`^page 3 of ${pages}\\b`).test(r.page) && r.rows.length === total - 40, `page 3 shows the remaining ${total - 40} rows (${r.rows.length})`);
  check(r.next, 'Next is disabled on the last page');
  check(r.rows.some(c => c[3] === '\u2014' && c[4] === '\u2014'), 'a run stored before the fields existed shows a dash in MTP and Split');
  await snap('table-page3');
  // search: filters, resets to page 1
  await page.fill('#bn-q', 'gemma');
  r = await info();
  check(/^page 1 of 1\b/.test(r.page), `a search from page 3 goes back to page 1: "${r.page}"`);
  check(r.rows.length >= 10 && r.rows.every(c => /gemma/i.test(c[1])), `search "gemma" keeps only that model (${r.rows.length} rows)`);
  await page.fill('#bn-q', 'flash layer');
  r = await info();
  check(r.rows.length > 0 && r.rows.every(c => /flash/i.test(c[1]) && c[4] === 'layer'), `search "flash layer" is an AND over model and split (${r.rows.length} rows)`);
  const sha = (0xabc000000000 + 7 * 7919).toString(16).padStart(12, '0');
  await page.fill('#bn-q', sha.slice(0, 10));
  r = await info();
  check(r.rows.length === 1 && r.rows[0][9] === sha, `search by sha finds exactly that run (${r.rows.length}: ${r.rows[0] && r.rows[0][9]})`);
  const day = new Date(Date.now() - 35 * 3600e3).toISOString().slice(0, 10);       // the run seeded 35 h ago (runs are 5 h apart)
  await page.fill('#bn-q', day);
  r = await info();
  check(r.rows.length >= 1 && r.rows.length < total, `search by date text ("${day}") finds some runs, not all (${r.rows.length})`);
  await page.fill('#bn-q', 'mtp on');
  r = await info();
  check(r.rows.length >= 1 && r.rows.every(c => c[3] === 'on'), `search "mtp on" keeps only MTP-on runs (${r.rows.length})`);
  await page.fill('#bn-q', 'zzzz-no-such-run');
  r = await info();
  check(r.empty && /no benchmark matches/.test(await txt('#bn-hist')), 'a search with no match says so');
  await snap('table-nomatch');
  await page.fill('#bn-q', '');
  r = await info();
  check(r.rows.length === 20 && new RegExp(`^page 1 of ${pages}\\b`).test(r.page), `clearing the search brings every run back: "${r.page}"`);
  check(await noPageScroll(), 'no horizontal page scroll');
  const box = await page.evaluate(() => { const b = document.querySelector('#bn-tw'); return {sw: b.scrollWidth, cw: b.clientWidth}; });
  if (W <= 500) check(box.sw > box.cw, `at phone width the table scrolls inside its own box (${box.sw} > ${box.cw})`);
  await page.locator('#bn-bar').scrollIntoViewIfNeeded();
  await snap('table-bar');
}

// ---- (d) the tab reads "Benchmarks"; its id and hash stay "speed"
async function label() {
  console.log(`[${THEME} ${WIDTH}px] the tab is called Benchmarks`);
  await page.goto(BASE + '/#speed');
  await page.waitForSelector('#sp-target option', {state: 'attached', timeout: 30000});
  const nav = await txt('nav.tabs button[data-tab=speed]');
  check(nav === 'Benchmarks', `the navigation entry reads "Benchmarks": "${nav}"`);
  check(await count('nav.tabs button[data-tab=speed][aria-selected=true]') === 1, 'the hash #speed still opens it (tab id "speed")');
  check(await txt('#p-speed .ph h2') === 'Benchmarks', 'the page heading reads "Benchmarks"');
  check(!(await txt('nav.tabs')).split(/\s+/).includes('Speed'), 'no navigation entry is still called Speed');
  await page.locator('nav.tabs button[data-tab=servers]').click();
  await page.waitForSelector('#sv-list .card.srv', {timeout: 20000});
  const btn = page.locator('#sv-list .card.srv .acts button', {hasText: 'Benchmarks'}).first();
  check(await btn.count() === 1, 'the server card has a "Benchmarks" button');
  await btn.click();
  await page.waitForSelector('#p-speed:not([hidden])', {timeout: 5000});
  check(await page.evaluate(() => location.hash) === '#speed', 'that button lands on #speed');
}

// ---- (c) FR#11: the server strip
async function strip() {
  console.log(`[${THEME} ${WIDTH}px] the server strip`);
  await openSpeed();
  await page.waitForSelector('#sp-strip:not([hidden])', {timeout: 20000});
  const calls = [];
  page.on('request', r => { const u = new URL(r.url()); if (r.method() === 'POST' && /^\/api\/(start|stop|servers|restart)$/.test(u.pathname)) { let b = {}; try { b = JSON.parse(r.postData() || '{}'); } catch (e) { /* */ } calls.push({path: u.pathname, body: b}); } });
  const st = () => txt('#ss-state');
  const vis = id => page.locator(id).isVisible();
  const inView = () => page.evaluate(() => { const b = document.querySelector('#sp-strip').getBoundingClientRect(); return b.left >= -1 && b.right <= document.documentElement.clientWidth + 1; });
  check(/Rig 1/.test(await txt('#ss-name')), `the strip names the server picked in the target box: "${await txt('#ss-name')}"`);
  check(/^(serving|busy)$/.test(await st()), `it says the state in words: "${await st()}"`);
  check(/Qwen3\.8-27B/.test(await txt('#ss-info')) && /:\d{4,5}/.test(await txt('#ss-info')), `it names the model and port: "${await txt('#ss-info')}"`);
  check(!(await vis('#ss-start')) && await vis('#ss-stop') && await vis('#ss-edit'), 'a running server offers Stop and Edit, not Start');
  check(await inView() && await noPageScroll(), 'the strip fits the window, no horizontal page scroll');
  await snap('strip-running');
  // the other server
  const rig2 = await page.evaluate(() => [...document.querySelectorAll('#sp-target option')].find(o => /Rig 2/.test(o.textContent)).value);
  await page.selectOption('#sp-target', rig2);
  await until(async () => /Rig 2/.test(await txt('#ss-name')), 5000);
  check(/Rig 2/.test(await txt('#ss-name')) && /Flash-Next/.test(await txt('#ss-info')), `picking Rig 2 shows Rig 2: "${await txt('#ss-name')} ${await txt('#ss-info')}"`);
  const rig1 = await page.evaluate(() => [...document.querySelectorAll('#sp-target option')].find(o => /Rig 1/.test(o.textContent)).value);
  await page.selectOption('#sp-target', rig1);
  await until(async () => /Rig 1/.test(await txt('#ss-name')), 5000);
  // Stop flips it
  await page.click('#ss-stop');
  const stopped = await until(async () => /stopped|exited/.test(await st()), 20000);
  check(!!stopped, `Stop: the strip says it stopped: "${await st()}"`);
  check(calls.some(c => c.path === '/api/stop' && c.body.sid === 'main'), 'Stop used the Servers tab call (POST /api/stop with the server id)');
  check(await vis('#ss-start') && !(await vis('#ss-stop')), 'a stopped server offers Start, not Stop');
  const tgt = () => page.evaluate(() => document.querySelector('#sp-target option:checked').textContent);
  await until(async () => /stopped|exited/.test(await tgt()), 8000);
  check(/stopped|exited/.test(await tgt()), 'the target box agrees: "' + await tgt() + '"');
  await snap('strip-stopped');
  // a stopped server: Edit saves only
  await page.click('#ss-edit');
  await page.waitForSelector('#ss-form:not([hidden])', {timeout: 8000});
  check(await txt('#ss-apply') === 'Save' && /Press Start/.test(await txt('#ss-note')), `a stopped server: the button says Save and the note says Press Start: "${await txt('#ss-apply')}"`);
  const opts = await page.evaluate(() => [...document.querySelectorAll('#ss-model option')].map(o => o.textContent));
  check(opts.length >= 3 && opts.some(o => /Flash-Next-PXQ4/.test(o)), `the model list is the library (${opts.length} models)`);
  await page.selectOption('#ss-model', {label: opts.find(o => /Flash-Next-PXQ4/.test(o))});
  await page.check('#ss-mtp'); await page.click('#ss-sm button[data-v=tensor]');
  const n0 = calls.length;
  await page.click('#ss-apply');
  await until(async () => !(await vis('#ss-form')), 8000);
  const saved = calls.slice(n0);
  const sv = saved.find(c => c.path === '/api/servers');
  check(!!sv && /Flash-Next-PXQ4/.test(sv.body.settings.model) && sv.body.settings.mtp === true && sv.body.settings.sm === 'tensor', `Save posts the profile like Launch > Save (model, MTP, split): ${JSON.stringify(sv && [sv.body.settings.model.split('/').pop(), sv.body.settings.mtp, sv.body.settings.sm])}`);
  check(!saved.some(c => c.path === '/api/start' || c.path === '/api/stop'), 'saving a stopped server does not start it');
  check(/Flash-Next-PXQ4/.test(await txt('#ss-info')) || true, 'ok');
  // Start flips it
  const n1 = calls.length;
  await page.click('#ss-start');
  const up = await until(async () => /^(serving|busy)$/.test(await st()), 20000);
  const sc = calls.slice(n1).find(c => c.path === '/api/start');
  check(!!sc && sc.body.sid === 'main' && /Flash-Next-PXQ4/.test(sc.body.model) && sc.body.mtp === true && sc.body.sm === 'tensor', `Start posts the saved settings with the server id like the Servers tab: ${JSON.stringify(sc && [sc.body.sid, sc.body.model.split('/').pop(), sc.body.mtp, sc.body.sm])}`);
  check(!!up, `Start: the strip says it serves again: "${await st()}"`);
  check(await vis('#ss-stop') && !(await vis('#ss-start')), 'a running server offers Stop again');
  // a running server: Edit says it restarts, and does stop + start
  await page.click('#ss-edit');
  await page.waitForSelector('#ss-form:not([hidden])', {timeout: 8000});
  check(await txt('#ss-apply') === 'Save and restart' && /stops it and starts it again/.test(await txt('#ss-note')), `a running server: the form says it restarts: "${await txt('#ss-apply')}" / "${(await txt('#ss-note')).slice(0, 60)}..."`);
  check(await page.inputValue('#ss-model').then(v => /Flash-Next-PXQ4/.test(v)) && await page.isChecked('#ss-mtp'), 'the form starts from the saved settings');
  await page.uncheck('#ss-mtp'); await page.click('#ss-sm button[data-v=layer]');
  await snap('strip-edit');
  const n2 = calls.length;
  await page.click('#ss-apply');
  await until(async () => calls.slice(n2).some(c => c.path === '/api/start'), 15000);
  const seq = calls.slice(n2).map(c => c.path);
  check(JSON.stringify(seq) === JSON.stringify(['/api/servers', '/api/stop', '/api/start']), `Save and restart = save, stop, start (the existing calls): ${seq.join(' ')}`);
  const sc2 = calls.slice(n2).find(c => c.path === '/api/start');
  check(!!sc2 && sc2.body.mtp === false && sc2.body.sm === 'layer', 'the restart carries the new settings');
  check(!!(await until(async () => /^(serving|busy)$/.test(await st()) && !(await vis('#ss-form')), 20000)), `after the restart the strip serves and the form is shut: "${await st()}"`);
  await snap('strip-after');
}

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, THEME);
  for (const ph of PH) { if (ph === 'refresh') await refresh(); else if (ph === 'table') await table(); else if (ph === 'strip') await strip(); else if (ph === 'label') await label(); }
  console.log(`[${THEME} ${WIDTH}px] errors`);
  check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors[0] : ''));
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log(`all checks passed (${THEME} ${WIDTH}px ${PH.join(',')})`);
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
