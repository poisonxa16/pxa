// Headless-browser check of two PXA Control fixes. Run it with tests/update-hs-gui-check.py (hermetic env, fake pxa-update, fake board):
//     node update-hs-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME PHASES
// PHASES is a comma list of:
//   hs      FR#13: Submit closes the "New high score" frame like Not now and shows the board's answer as a toast; a sent result never
//           brings the frame or the Check button back (a reload included); a new run hides an open frame
//   update  FR#14: Update asks again on every open; no update = "Up to date" (running servers or not); update + nothing running =
//           install; update + a running server = stop first. The page loads with two servers running, then they stop.
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, PHASES] = process.argv.slice(2);
const PH = (PHASES || 'hs,update').split(',');
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name, sel) {
  shot++;
  const path = `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`;
  if (sel) await page.locator(sel).screenshot({path}); else await page.screenshot({path, fullPage: false});
}
const txt = async sel => (await page.locator(sel).allInnerTexts()).join(' | ').replace(/\s+/g, ' ').trim();
const vis = sel => page.locator(sel).isVisible();
async function until(fn, ms, step) { const t0 = Date.now(); for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t0 > ms) return v; await sleep(step || 250); } }
const call = (path, opts) => page.evaluate(([p, o]) => api(p, o), [path, opts || {}]);
const toastText = () => page.evaluate(() => { const t = document.getElementById('toast'); return t && !t.hidden ? t.textContent : ''; });
const noPageScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);

async function openSpeed() {
  await page.goto(BASE + '/#speed');
  await page.waitForSelector('#sp-target option', {state: 'attached', timeout: 30000});
  await sleep(1200);
}
async function runBench() {
  await page.fill('#bn-warm', '0'); await page.fill('#bn-reps', '1');
  await page.click('#bn-run');
  const running = await until(async () => /running/.test(await txt('#bn-run')), 8000, 100);
  const fin = await until(async () => /Run benchmark/.test(await txt('#bn-run')) && /done/.test(await txt('#bn-lines')), 90000, 300);
  return {running: !!running, fin: !!fin};
}

// ---- FR#13
async function hs() {
  console.log(`[${THEME} ${WIDTH}px] FR#13: the high-score frame after Submit`);
  await openSpeed();
  let r = await runBench();
  check(r.fin, 'benchmark 1 finished');
  check(!!(await until(() => vis('#hs-check'), 8000)), 'after the run: "Check the community high-score board" is offered');
  await page.click('#hs-check');
  check(!!(await until(() => vis('#hs'), 15000)), 'Check: the "New high score" frame opens (the fake board has no record)');
  await page.fill('#hs-name', 'Tester');
  await page.locator('#hs').scrollIntoViewIfNeeded();
  await snap('hs-frame-before-submit');
  await page.click('#hs-send');
  const closed = await until(async () => !(await vis('#hs')), 8000, 100);
  const t1 = await until(toastText, 3000, 50);
  check(!!closed, 'Submit closes the frame, the same way Not now does');
  check(/^On the board at rank 1\.$/.test(t1 || ''), `the board's answer shows as a toast: "${t1}"`);
  check(!(await vis('#hs-check')), 'the Check button is gone for the sent result');
  await snap('hs-after-submit');
  await sleep(3000);
  check(!(await vis('#hs')) && !(await vis('#hs-check')), 'three seconds later: still closed, no Check button');
  await page.reload();
  await page.waitForSelector('#sp-target option', {state: 'attached', timeout: 30000});
  await sleep(2500);
  check(!(await vis('#hs')) && !(await vis('#hs-check')), 'after a reload: the sent result brings back neither the frame nor the Check button');
  const sc = await call('/api/score/check');
  check(sc.available === false && /already on the board/.test(sc.reason || ''), `Control refuses to offer the sent result again: ${JSON.stringify(sc.reason)}`);
  await snap('hs-after-reload');
  // a new run is a new result: it may be offered; an open frame hides once the next run starts
  r = await runBench();
  check(r.fin, 'benchmark 2 finished');
  check(!!(await until(() => vis('#hs-check'), 8000)), 'the new result is offered (it was never sent)');
  await page.click('#hs-check');
  check(!!(await until(() => vis('#hs'), 15000)), 'the frame opens for the new result');
  await page.fill('#bn-warm', '0'); await page.fill('#bn-reps', '1');
  await page.click('#bn-run');
  const hid = await until(async () => !(await vis('#hs')), 6000, 100);
  check(!!hid, 'starting benchmark 3 hides the frame of result 2 (it no longer stays up across runs)');
  check(!!(await until(async () => /Run benchmark/.test(await txt('#bn-run')) && /done/.test(await txt('#bn-lines')), 90000, 300)), 'benchmark 3 finished');
  await until(() => vis('#hs-check'), 8000);
  await page.click('#hs-check');
  check(!!(await until(() => vis('#hs'), 15000)), 'the frame opens for result 3');
  await page.click('#hs-send');
  const closed2 = await until(async () => !(await vis('#hs')), 8000, 100);
  const t2 = await until(toastText, 3000, 50);
  check(!!closed2 && /^Received\. It is held for a quick review before it shows on the board\.$/.test(t2 || ''), `a submit held for review also closes the frame and says so: "${t2}"`);
  check(await noPageScroll(), 'no sideways page scroll');
}

// ---- FR#14
async function openDlg() {
  if (await page.evaluate(() => document.getElementById('up').open)) { await page.click('#up-cancel'); await sleep(200); }
  await page.evaluate(() => { const b = [...document.querySelectorAll('#hm-up, #sv-up')].find(x => !x.hidden) || document.getElementById('hm-up'); b.click(); });
  await until(async () => !/Checking/.test(await txt('#up-sub')), 15000, 100);
  await sleep(300);
  return {sub: await txt('#up-sub'), body: await txt('#up-body'), track: await vis('#up-track'),
          blocked: await page.evaluate(() => document.getElementById('up-track').classList.contains('is-blocked'))};
}
async function update() {
  console.log(`[${THEME} ${WIDTH}px] FR#14: the Update dialog`);
  await page.goto(BASE + '/');
  await until(() => page.evaluate(() => !!(S.update && S.update.binary)), 20000);
  const sv = (await call('/api/servers')).servers;
  const n0 = await page.evaluate(() => S.update.servers_running);
  check(n0 >= 1, `the page loaded while ${n0} server(s) ran (its first count)`);
  for (const s of sv) if (s.running) await call('/api/stop', {body: {sid: s.sid}, timeout: 60000});
  const left = (await call('/api/servers')).servers.filter(s => s.running).length;
  check(left === 0, `every server is stopped now (${left} running)`);
  // case 1: no update, nothing running (the report: the page still said 'Stop the running servers first')
  let d = await openDlg();
  check(!/Stop the running servers/.test(d.body), `no update + nothing running: no stop line: "${d.body}"`);
  check(/^Up to date\./.test(d.body) && /Up to date/.test(d.sub), `no update + nothing running: says up to date: "${d.sub}" / "${d.body}"`);
  check(!d.track, 'no update: no install slider');
  await snap('update-1-uptodate', '#up');
  // case 4: no update while a server runs: still up to date (the stop line belongs only to a waiting update)
  await call('/api/start', {body: {sid: 'main'}});
  d = await openDlg();
  check(/^Up to date\./.test(d.body) && !/Stop the running/.test(d.body), `no update + a server running: up to date, no stop line: "${d.body}"`);
  await call('/api/stop', {body: {sid: 'main'}, timeout: 60000});
  // case 2: an update waits, nothing running
  await call('/api/update/channel', {method: 'POST', body: {channel: 'beta'}, timeout: 70000});
  d = await openDlg();
  check(/^Install 4\.0-beta\.6\./.test(d.body) && /4\.0-beta\.6 available/.test(d.sub) && !/running/.test(d.sub), `update + nothing running: offers the install: "${d.sub}" / "${d.body}"`);
  check(d.track && !d.blocked, 'update + nothing running: the slider is there and not blocked');
  await snap('update-2-install', '#up');
  // case 3: an update waits and a server runs
  await call('/api/start', {body: {sid: 'main'}});
  d = await openDlg();
  check(/^Stop the running servers first\. PXA will not restart them\.$/.test(d.body) && /1 running/.test(d.sub), `update + one running: stop first: "${d.sub}" / "${d.body}"`);
  check(d.track && d.blocked, 'update + one running: the slider is blocked');
  await snap('update-3-stopfirst', '#up');
  // the guard on the server agrees
  const code = await page.evaluate(async () => { try { await api('/api/update/apply', {method: 'POST', body: {}}); return 'installed'; } catch (e) { return e.code; } });
  check(code === 'servers_running', `the install route refuses while the server runs (${code})`);
  await call('/api/stop', {body: {sid: 'main'}, timeout: 60000});
  d = await openDlg();
  check(/^Install 4\.0-beta\.6\./.test(d.body), `after Stop, the next open offers the install again: "${d.body}"`);
  await page.click('#up-cancel');
  check(await noPageScroll(), 'no sideways page scroll');
}

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: +WIDTH, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, THEME);
  for (const ph of PH) { if (ph === 'hs') await hs(); else if (ph === 'update') await update(); }
  console.log(`[${THEME} ${WIDTH}px] errors`);
  const real = errors.filter(e => !/status of 409/.test(e));        // the refused install above is a 409 the browser logs
  check(real.length === 0, 'no page error' + (real.length ? ': ' + real[0] : ''));
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log(`all checks passed (${THEME} ${WIDTH}px ${PH.join(',')})`);
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
