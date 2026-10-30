// Headless-browser check of the Live tab's "Running only" switch (FR#10: stopped servers should not pile up as tiles).
// Run it with tests/live-stopped-gui-check.py (it starts three fake servers and the browser):
//     node live-stopped-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME
// The switch is on by default and remembered on the machine; with it on, stopped servers' tiles fold into ONE line
// "N stopped (show)" that opens them again; off = every tile, as before. It stops servers of the environment on the way.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-live-${name}.png`, fullPage: true}); }
const tiles = () => page.locator('#lv-tiles .lv-tile').count();
const names = async () => (await page.locator('#lv-tiles .lv-name').allInnerTexts()).join('|');
const fold = async () => (await page.locator('#lv-stopped').isVisible()) ? (await page.locator('#lv-stopped').innerText()).replace(/\s+/g, ' ').trim() : null;
async function until(fn, ms) { for (let t = 0; t < (ms || 20000); t += 250) { if (await fn()) return true; await sleep(250); } return false; }
async function open() {
  await page.goto(BASE + '/#live');
  await page.waitForSelector('#lv-tiles .lv-tile', {timeout: 30000});
  await sleep(1500);
}
(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, THEME);
  await open();
  const sid = Object.fromEntries(await page.evaluate(async () => (await api('/api/servers')).servers.map(s => [s.name, s.sid])));
  const stop = n => page.evaluate(s => api('/api/stop', {body: {sid: s}, timeout: 60000}), sid[n]);

  console.log(`[${THEME} ${WIDTH}px] default`);
  check(await page.locator('#lv-running').isChecked(), 'the "Running only" switch is on by default');
  check(/Running only/.test(await page.locator('#lv-running-wrap').innerText()), 'the switch is named "Running only"');
  check(await tiles() === 3 && await fold() === null, `three running servers: three tiles, no fold line (${await tiles()}, ${await fold()})`);
  await snap('running');

  console.log(`[${THEME} ${WIDTH}px] one stops`);
  await stop('Rig 2');
  check(await until(async () => (await fold()) === '1 stopped (show)'), 'the stopped server folds into "1 stopped (show)": ' + await fold());
  check(await tiles() === 2 && !/Rig 2/.test(await names()), `two tiles are left, Rig 2 is not one of them (${await names()})`);
  check(await page.locator('#lh-dec .lv-legend span').count() >= 2, 'the charts still carry every server (a finished run stays reachable)');
  await snap('one-folded');
  await page.locator('#lv-stopped-btn').click(); await sleep(600);
  check(await tiles() === 3 && /Rig 2/.test(await names()), `show brings the stopped tile back (${await names()})`);
  check(await fold() === '1 stopped (hide)', 'the line now offers (hide): ' + await fold());
  const chip = await page.locator('#lv-tiles .lv-tile', {hasText: 'Rig 2'}).locator('.chip').innerText();
  check(/stopped/.test(chip), 'the stopped tile says "stopped": ' + chip);
  await snap('one-shown');
  await page.locator('#lv-stopped-btn').click(); await sleep(400);
  check(await tiles() === 2 && await fold() === '1 stopped (show)', 'hide folds it again');

  console.log(`[${THEME} ${WIDTH}px] the switch off`);
  await page.locator('#lv-running-wrap').click(); await sleep(500);
  check(!(await page.locator('#lv-running').isChecked()), 'the switch is off');
  check(await tiles() === 3 && await fold() === null, `off: every tile is back and there is no fold line (${await tiles()}, ${await fold()})`);
  await snap('switch-off');
  await open();
  check(!(await page.locator('#lv-running').isChecked()) && await tiles() === 3, 'off survives a reload');
  await page.locator('#lv-running-wrap').click(); await sleep(500);
  check(await page.locator('#lv-running').isChecked() && await tiles() === 2 && await fold() === '1 stopped (show)', 'on again: folded');
  await open();
  check(await page.locator('#lv-running').isChecked() && await fold() === '1 stopped (show)', 'on survives a reload');

  console.log(`[${THEME} ${WIDTH}px] many stopped`);
  await stop('Rig 1'); await stop('Rig 3');
  check(await until(async () => (await fold()) === '3 stopped (show)'), 'three stopped servers are still ONE line: ' + await fold());
  check(await tiles() === 0 && /No server is running/.test(await page.locator('#lv-tiles').innerText()), 'with nothing running the tiles area says so');
  await snap('all-folded');
  await page.locator('#lv-stopped-btn').click(); await sleep(600);
  check(await tiles() === 3, `show opens all three (${await tiles()})`);
  await snap('all-shown');

  const w = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(w.sw <= w.cw + 1, `no horizontal page scroll (${w.sw} <= ${w.cw})`);
  check(errors.length === 0, 'no page error or console error' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await b.close();
  console.log(failures.length ? `FAILED ${failures.length}: ` + failures.join('; ') : 'ALL PASSED');
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('crash: ' + e.stack); process.exit(2); });
