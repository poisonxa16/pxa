// Headless-browser check of the tokens-per-watt column and chart on PXA Control's Benchmarks tab. Run it with tests/watt-gui-check.py:
//     node watt-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME PHASES
// PHASES is a comma list of: table, chart, run, na.
//   table  the history table has the avg W / peak W / tok/s per W columns; a seeded row shows watts and decode/avg W; the rows stored before
//          the feature (and one with an empty summary) show a dash, never NaN / Infinity / undefined; paging (20 per page) and the search
//          still work; no horizontal page scroll; at phone width the table scrolls inside its own box
//   chart  the tokens-per-watt chart draws (canvas ink, legend with the median), follows the search, says so when no run has a reading,
//          fits the window
//   run    a benchmark run through the GUI against the fake power feed: the log and the line under the run show average / peak W and
//          tokens per second per watt, the new newest row carries them (prose decode / average W), and the number is sane
//   na     the same on a host whose nvidia-smi answers [N/A] for the power: no power line, a dash on the new row, no NaN, no error
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, PHASES] = process.argv.slice(2);
const W = +WIDTH, PH = (PHASES || 'table,chart,run').split(',');
const DASH = '—';
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
const txt = async sel => (await page.locator(sel).allInnerTexts()).join(' | ').replace(/\s+/g, ' ').trim();
async function until(fn, ms, step) { const t0 = Date.now(); for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t0 > ms) return v; await sleep(step || 250); } }
const noPageScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
const ink = id => page.evaluate(i => { const c = document.getElementById(i); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0, h = 0; for (let k = 0; k < d.length; k += 4) if (d[k + 3]) { n++; h = (h * 31 + d[k] + 7 * d[k + 1] + 13 * d[k + 2]) | 0; } return {ink: n, sig: h, w: c.clientWidth, h: c.clientHeight}; }, id);
const fits = sel => page.evaluate(s => { const b = document.querySelector(s).getBoundingClientRect(); return b.left >= -1 && b.right <= document.documentElement.clientWidth + 1; }, sel);
const rowsInfo = () => page.evaluate(() => ({
  rows: [...document.querySelectorAll('#bn-hist tr')].filter(tr => !tr.querySelector('.empty')).map(tr => [...tr.children].map(td => td.textContent)),
  page: document.querySelector('#bn-page').textContent}));
const BAD = /NaN|Infinity|undefined|null/;
async function openSpeed() {
  await page.goto(BASE + '/#speed');
  await page.waitForSelector('#sp-target option', {state: 'attached', timeout: 30000});
  await page.waitForSelector('#bn-hist tr td:not(.empty)', {timeout: 20000});
  await sleep(1500);
}
const COL = {prose: 5, avg: 10, peak: 11, tpw: 12};

async function table() {
  console.log(`[${THEME} ${WIDTH}px] the history table: tokens per watt column, dashes, paging, search`);
  await openSpeed();
  const heads = await txt('table.hist thead');
  check(/avg W/i.test(heads) && /peak W/i.test(heads) && /tok\/s per W/i.test(heads), `the table has the avg W, peak W and tok/s per W columns: ${heads.slice(-40)}`);
  let r = await rowsInfo();
  check(r.rows.length === 20 && r.rows.every(c => c.length === 13), `page 1 shows 20 rows of 13 cells (${r.rows.length})`);
  const withPow = r.rows.filter(c => c[COL.tpw] !== DASH);
  check(withPow.length >= 18, `the newest runs show a power reading (${withPow.length} of 20; the one with an empty summary shows a dash)`);
  const sane = withPow.every(c => { const p = +c[COL.prose], a = +c[COL.avg], pk = +c[COL.peak], t = +c[COL.tpw]; return a >= 100 && a <= 400 && pk > a && Math.abs(t - p / a) < 0.0011 && t > 0.05 && t < 1; });
  check(sane, `tok/s per W = prose decode / average W, peak above average, watts in range: ${JSON.stringify(withPow[0])}`);
  check(!BAD.test(await txt('#bn-hist')), 'no NaN / Infinity / undefined / null in the table');
  const tip = await page.evaluate(() => { const td = [...document.querySelectorAll('#bn-hist tr')].map(tr => tr.children[10]).find(td => td && td.textContent !== '—'); return td ? td.title : ''; });
  check(/#0: avg .* W, peak .* W; #1/.test(tip), `the watt cells carry a per-card tooltip: "${tip.slice(0, 60)}"`);
  await snap('table-page1');
  await page.click('#bn-next');
  r = await rowsInfo();
  const shaOf = i => (0xabc000000000 + i * 7919).toString(16).padStart(12, '0');
  const rowOf = i => r.rows.find(c => c[9] === shaOf(i));
  check(r.rows.filter(c => c[COL.tpw] !== DASH).length === 9, `page 2: the 9 runs with a reading show numbers, the 10 older ones a dash (${r.rows.filter(c => c[COL.tpw] !== DASH).length} with numbers)`);
  const e = rowOf(20), nb = rowOf(21);
  check(e && e[COL.avg] === DASH && e[COL.peak] === DASH && e[COL.tpw] === DASH && nb && nb[COL.tpw] !== DASH, 'the run whose power summary is empty shows a dash in all three, its neighbour shows numbers');
  await page.click('#bn-next');
  r = await rowsInfo();
  check(/^page 3 of 3\b/.test(r.page) && r.rows.length === 5, `paging still works: page 3 shows the last 5 runs ("${r.page}")`);
  check(r.rows.every(c => c[COL.avg] === DASH && c[COL.peak] === DASH && c[COL.tpw] === DASH), 'runs stored before the feature show a dash, not 0 or NaN');
  check(!BAD.test(await txt('#bn-hist')), 'no NaN / Infinity / undefined / null on the old page either');
  await snap('table-page3-old');
  await page.fill('#bn-q', 'flash layer');
  r = await rowsInfo();
  check(/^page 1 of 1\b/.test(r.page) && r.rows.length > 0 && r.rows.every(c => /flash/i.test(c[1]) && c[4] === 'layer'), `search "flash layer" still filters and resets to page 1 (${r.rows.length} rows)`);
  await page.fill('#bn-q', '');
  r = await rowsInfo();
  check(r.rows.length === 20 && /^page 1 of 3\b/.test(r.page), `clearing the search brings every run back ("${r.page}")`);
  check(await noPageScroll(), 'no horizontal page scroll');
  const box = await page.evaluate(() => { const b = document.querySelector('#bn-tw'); return {sw: b.scrollWidth, cw: b.clientWidth}; });
  if (W <= 500) check(box.sw > box.cw, `at phone width the table scrolls inside its own box (${box.sw} > ${box.cw})`);
  else check(box.sw <= box.cw + 1, `at desktop width the 13 columns fit without scrolling the box (${box.sw} <= ${box.cw})`);
}

async function chart() {
  console.log(`[${THEME} ${WIDTH}px] the tokens-per-watt chart`);
  await openSpeed();
  await page.locator('#ch-watt').scrollIntoViewIfNeeded(); await sleep(300);
  const all = await ink('ch-watt');
  check(all.w > 100 && all.h >= 100 && all.ink > 1500, `the chart canvas is drawn (${all.w}x${all.h}, ${all.ink} px of ink)`);
  const lg = await txt('#lg-watt');
  check(/tokens\/s per watt/.test(lg) && /\d\.\d{3} median/.test(lg), `the legend names the series and its median: "${lg}"`);
  check(await fits('#ch-watt') && await noPageScroll(), 'the chart fits the window, no horizontal page scroll');
  await snap('chart');
  await page.fill('#bn-q', 'zzzz-no-such-run');
  await sleep(300);
  const none = await ink('ch-watt');
  check(none.sig !== all.sig && none.ink > 20 && none.ink < all.ink / 4, `with no run left the chart shows only its notice (${none.ink} px vs ${all.ink})`);
  check((await txt('#lg-watt')) === '', 'and the legend is empty');
  await page.fill('#bn-q', 'gemma');
  await sleep(300);
  const g = await ink('ch-watt');
  check(g.sig !== all.sig && g.ink > 1000, `the chart follows the search (gemma runs only: ${g.ink} px of ink, differs from the full chart)`);
  await page.fill('#bn-q', '');
  await sleep(300);
  const back = await ink('ch-watt');
  check(back.sig === all.sig, 'clearing the search restores the same chart');
  await page.setViewportSize({width: W === 390 ? 360 : 1100, height: 900}); await sleep(600);
  const rs = await ink('ch-watt');
  check(rs.ink > 1000 && await noPageScroll(), 'the chart redraws after a resize without a page scroll');
  await page.setViewportSize({width: W, height: 900}); await sleep(400);
  await page.locator('nav.tabs button[data-tab=servers]').click(); await sleep(500);
  await page.locator('nav.tabs button[data-tab=speed]').click(); await sleep(1500);
  const rt = await ink('ch-watt');
  check(rt.ink > 1500, `the chart is still drawn after leaving and re-opening the tab (${rt.ink} px)`);
}

async function runBench(expectPower) {
  await openSpeed();
  await page.fill('#bn-warm', '0'); await page.fill('#bn-reps', '1');
  await page.click('#bn-run');
  check(!!(await until(async () => /running/.test(await txt('#bn-run')), 8000)), 'the benchmark starts');
  const fin = await until(async () => /Run benchmark/.test(await txt('#bn-run')) && /done/.test(await txt('#bn-lines')), 90000, 400);
  check(!!fin, 'the benchmark finished');
  await sleep(1500);
  const log = await txt('#bn-lines'), now = await txt('#bn-watt-now');
  const r = await rowsInfo();
  const top = r.rows[0] || [];
  check(!/NaN|Infinity|undefined/.test(log + now + (await txt('#bn-hist'))), 'no NaN / Infinity / undefined in the log, the power line or the table');
  if (expectPower) {
    check(/power: average \d+ W, peak \d+ W, \d\.\d{3} tokens\/s per watt/.test(log), `the log has the power line: "${(log.match(/power: .*/) || [''])[0].slice(0, 90)}"`);
    check(/Power during this run: average \d+ W, peak \d+ W, \d\.\d{3} tokens\/s per watt \(\d+ samples\)/.test(now), `the line under the run: "${now}"`);
    const p = +top[COL.prose], a = +top[COL.avg], pk = +top[COL.peak], t = +top[COL.tpw];
    check(top.length === 13 && a > 150 && a < 800 && pk >= a && t > 0.03 && t < 1 && Math.abs(t - p / a) < Math.max(0.002, 0.03 * t), `the new newest row: prose ${p} t/s, ${a} W avg, ${pk} W peak, ${t} tok/s per W (= prose / avg W)`);
    const j = await page.evaluate(async () => (await (await fetch('/api/bench')).json()).history.slice(-1)[0].power);
    check(j && j.samples >= 1 && j.cards.length === 2 && j.cards.every(c => c.avg_w > 100 && c.limit_w === 250), `the stored row carries the summary: ${j && j.samples} samples, cards ${j && j.cards.map(c => c.index + ':' + c.avg_w + 'W').join(' ')}`);
    await page.locator('#ch-watt').scrollIntoViewIfNeeded(); await sleep(300);
    const c = await ink('ch-watt');
    check(c.ink > 1500, `the chart still draws with the new run (${c.ink} px)`);
    await page.locator('#bn-lines').scrollIntoViewIfNeeded();
    await snap('after-run-power');
  } else {
    check(!/power:/.test(log), 'the log has no power line when the card reports no draw');
    check(now === '', `no "Power during this run" line: "${now}"`);
    check(top.length === 13 && top[COL.avg] === DASH && top[COL.peak] === DASH && top[COL.tpw] === DASH && +top[COL.prose] > 0, `the new newest row shows a dash for watts (prose ${top[COL.prose]} t/s): ${JSON.stringify(top.slice(COL.prose, 13))}`);
    const j = await page.evaluate(async () => (await (await fetch('/api/bench')).json()).history.slice(-1)[0].power);
    check(j && j.avg_w === null && j.peak_w === null && j.tps_per_w === null && j.samples === 0, `the stored row has null power fields, not NaN: ${JSON.stringify(j)}`);
    await snap('after-run-na');
  }
  check(await noPageScroll(), 'no horizontal page scroll');
}
async function run() { console.log(`[${THEME} ${WIDTH}px] a benchmark with a power feed`); await runBench(true); }
async function na() { console.log(`[${THEME} ${WIDTH}px] a host whose nvidia-smi reports no power`); await runBench(false); }

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, THEME);
  for (const ph of PH) { if (ph === 'table') await table(); else if (ph === 'chart') await chart(); else if (ph === 'run') await run(); else if (ph === 'na') await na(); }
  console.log(`[${THEME} ${WIDTH}px] errors`);
  check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors[0] : ''));
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log(`all checks passed (${THEME} ${WIDTH}px ${PH.join(',')})`);
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
