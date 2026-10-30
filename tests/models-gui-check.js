// Headless-browser check of the Models tab (FR#3: folders first, the PXA list on demand).
// Run it with tests/models-gui-check.py, which starts PXA Control against a fake huggingface.co and the browser:
//     node models-gui-check.js BASE_URL ORIGIN_URL VIEWPORT_WIDTH OUT_DIR THEME PHASE
// PHASE main    : a fresh remembered list (read 40 min ago) -> a visit and a reload make no network call, the label says
//                 "checked 40 min ago", the button reads huggingface.co (force), filter / Download / Unlock still work
//         stale : remembered list 30 h old -> the visit re-reads on its own
//         nomodel: no model in the folders (first run) -> the visit reads on its own
//         nocache: nothing remembered yet -> the visit reads on its own, once
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, ORIGIN, WIDTH, OUT, THEME, PHASE] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [], reqs = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${PHASE}-${name}.png`, fullPage: true}); }
const listings = async () => (await (await fetch(ORIGIN + '/_listings')).json()).listings;
const txt = async sel => (await page.locator(sel).first().innerText()).replace(/\s+/g, ' ').trim();
const avail = () => reqs.filter(u => u.includes('/api/models/available'));
async function rows() { await page.waitForFunction(() => /Pub-Model-PXQ4\.gguf/.test(document.querySelector('#hf-body').innerText), null, {timeout: 30000}); }
async function visit() {
  await page.goto(BASE + '/#models');
  await page.waitForSelector('#models-body', {timeout: 30000});
  await rows();
  await sleep(1500);
}

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('request', r => reqs.push(r.url()));
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); localStorage.removeItem('pxa-control.hfFilter'); } catch (e) { /* */ } }, THEME);
  const n0 = await listings();
  console.log(`[${THEME} ${WIDTH}px ${PHASE}] visit the Models tab`);
  await visit();
  const n1 = await listings();

  if (PHASE !== 'main') {
    check(n1 - n0 >= 1, `the visit read huggingface.co on its own (${n1 - n0} listing request(s))`);
    if (PHASE === 'nocache') check(n1 - n0 === 1, `...once (${n1 - n0})`);
    check(/checked just now/.test(await txt('#hf-checked')), 'the label says "checked just now": ' + await txt('#hf-checked'));
    if (PHASE === 'nomodel') check(await page.locator('#models-body .state, #models-body .empty').count() >= 1, 'the library says there is no model yet');
    await snap('visit');
    check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.join(' | ') : ''));
    await b.close();
    console.log(failures.length ? `FAILED ${failures.length}: ` + failures.join('; ') : 'ALL PASSED');
    process.exit(failures.length ? 1 : 0);
  }

  // ---- main ----
  const y = sel => page.locator(sel).first().evaluate(e => Math.round(e.getBoundingClientRect().top + scrollY));
  const yf = await y('#folders-card'), yl = await y('#models-body'), yh = await y('#hf-card');
  check(yf < yl && yl < yh, `order: model folders (${yf}) above the Library (${yl}) above Available from PXA (${yh})`);
  check(await page.locator('#folders-card #dirs').count() === 1 && await page.locator('#folders-card #dir-add').count() === 1
    && await page.locator('#folders-card #models-rescan').count() === 1, 'the folders card holds the folder list, Add folder and Rescan');
  check(await page.locator('#models-body tr.click').count() >= 1, 'the library lists the local model');
  check(n1 - n0 === 0, `a visit with a fresh remembered list makes no huggingface.co request (${n1 - n0})`);
  check(avail().length >= 1 && avail().every(u => /cached=1/.test(u) && !/force=1/.test(u)), 'the page asked the server for the remembered list only: ' + avail().map(u => u.replace(BASE, '')).join(' '));
  check(/^checked 4\d min ago$/.test(await txt('#hf-checked')), 'the list says how old it is: "' + await txt('#hf-checked') + '"');
  check(/3 model file\(s\) in 3 repositories/.test(await txt('#hf-sub')), 'the remembered list is shown: ' + await txt('#hf-sub'));
  check(await page.locator('#hf-refresh').innerText() === 'Check PXA models', 'the button reads "Check PXA models"');
  const sw = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(sw.sw <= sw.cw + 1, `no horizontal page scroll (${sw.sw} <= ${sw.cw})`);
  await snap('visit');

  console.log(`[${THEME} ${WIDTH}px ${PHASE}] reload keeps the list`);
  await visit();
  check(await listings() - n0 === 0, 'after a page reload there is still no huggingface.co request');
  check(/^checked 4\d min ago$/.test(await txt('#hf-checked')), 'the age survives the reload: "' + await txt('#hf-checked') + '"');
  // leave and come back inside the same page: still nothing
  await page.locator('nav.tabs button[data-tab=servers]').click(); await sleep(500);
  await page.locator('nav.tabs button[data-tab=models]').click(); await sleep(1500);
  check(await listings() - n0 === 0, 'leaving and re-entering the tab makes no request either');

  console.log(`[${THEME} ${WIDTH}px ${PHASE}] the list as it was`);
  const total = await page.locator('#hf-body tr').count();
  const filt = page.locator('#hf-filter button');
  check(await filt.count() >= 3, 'the tier filter is there: ' + (await filt.allInnerTexts()).join(' | '));
  await page.locator('#hf-filter button', {hasText: 'Free PXQ'}).click(); await sleep(300);
  const free = await page.locator('#hf-body tr').count();
  check(free > 0 && free < total, `the filter narrows the list (${total} -> ${free})`);
  await page.locator('#hf-filter button', {hasText: 'Everything'}).click(); await sleep(300);
  check(await page.locator('#hf-body button:has-text("Download")').count() >= 1, 'Download buttons are there');
  check(await page.locator('#hf-body button:has-text("Unlock")').count() >= 1, 'the supporter file carries its lock and Unlock');

  console.log(`[${THEME} ${WIDTH}px ${PHASE}] the button reads huggingface.co`);
  const before = await listings();
  const mark = reqs.length;
  await page.locator('#hf-refresh').click();
  await page.waitForFunction(() => /checked just now/.test(document.querySelector('#hf-checked').innerText), null, {timeout: 30000});
  check(await listings() - before >= 1, `Check PXA models made Control ask huggingface.co (${await listings() - before} listing request(s))`);
  check(reqs.slice(mark).some(u => /force=1/.test(u)), 'the click sent force=1');
  check(/3 model file\(s\)/.test(await txt('#hf-sub')), 'the list is shown again');
  await snap('after-check');

  check(errors.length === 0, 'no page error or console error' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await b.close();
  console.log(failures.length ? `FAILED ${failures.length}: ` + failures.join('; ') : 'ALL PASSED');
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('crash: ' + e.stack); process.exit(2); });
