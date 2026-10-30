// Headless-browser check of two owed chat-page items (todo control-chat-hardening), run by tests/chat-owed-gui-check.py:
//     node chat-owed-gui-check.js BASE_URL(with ?token=) WIDTH OUT_DIR THEME PHASE RAW_FILE_TEXT
// PHASE (one per run):
//   broken  a damaged chat file (planted by the runner) is listed as a marked "Broken" row with Export raw + Delete in view; clicking it
//           shows a friendly card (no error toast); a reload with that chat remembered shows the card too; raw export is byte-identical;
//           Delete (with the confirm) removes it
//   image   the server answers 400 bad_image for an unsupported image: the reason shows as a toast and a notice, the typed words are back
//           in the box, the bad chip can be removed, and the next send works
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, PHASE, RAW] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${PHASE}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: false}); }
async function until(fn, ms = 30000, arg) { try { await page.waitForFunction(fn, arg, {timeout: ms, polling: 200}); return true; } catch (e) { return false; } }
const body = () => page.evaluate(() => document.body.innerText);
const idle = () => until(() => { const b = document.querySelector('#ag-send'); return b && b.getAttribute('aria-label') === 'Send' && !b.disabled; }, 60000);
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

async function open(url) {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForSelector('#ag-root #ag-in', {timeout: 60000});
  await until(() => { const b = document.querySelector('#ag-send'); return b && !b.disabled; }, 60000);
}
async function showList() {         // at phone width the chat list is a drawer
  if (!(await page.locator('.ag-side').first().isVisible()) || W <= 900) {
    const sh = page.locator('#ag-side-show');
    if (await sh.count() && await sh.first().isVisible()) await sh.first().click();
  }
  await sleep(300);
}
const visible = async loc => { if (!(await loc.count())) return false; const b = await loc.first().boundingBox(); return !!b && b.width > 4 && b.height > 4 && await loc.first().isVisible(); };

async function broken() {
  console.log(`[${THEME} ${WIDTH}px] ST06: the broken-chat row`);
  await showList();
  const row = page.locator('.ag-item.broken[data-id="c_corrupt0001"]');
  check(await row.count() === 1, 'the damaged chat is listed as a broken row');
  check(await visible(row), 'the row is visible');
  const t = (await row.textContent()).replace(/\s+/g, ' ');
  console.log('       row text: ' + t);
  check(/Broken/.test(t) && /broken one/.test(t), 'it is marked "Broken" and keeps the saved title ("broken one")');
  check(!/could not be read/.test(t), 'the raw "(could not be read)" suffix is not shown twice');
  const exp = row.locator('a.ag-broken-b'), del = row.locator('button.ag-broken-b');
  check(await visible(exp) && await visible(del), 'Export raw and Delete are visible without hovering');
  const href = await exp.getAttribute('href');
  check(/format=raw/.test(href) && /id=c_corrupt0001/.test(href), 'Export raw points at the raw export (' + href + ')');
  const rawGot = await page.evaluate(async h => (await fetch(h)).text(), href);
  check(rawGot === RAW, 'the raw export is the file byte for byte');
  const bb = await row.boundingBox();
  check(bb.x >= 0 && bb.x + bb.width <= W + 1, 'the row fits the screen width (x ' + Math.round(bb.x) + ', w ' + Math.round(bb.width) + ')');
  await snap('list-broken-row');
  console.log('       opening it');
  await row.locator('.ag-item-t').click();
  check(await until(() => !!document.querySelector('#ag-broken-card'), 8000), 'a friendly card appears');
  const b = await body();
  check(/could not be opened/.test(b) && /damaged/.test(b) && /Nothing was deleted/.test(b), 'the card says what happened in words');
  const toastHidden = await page.evaluate(() => { const t = document.querySelector('#toast'); return !t || t.hidden || !t.textContent; });
  check(toastHidden, 'no error toast is shown');
  check(!/HTTP 422|session_unreadable/.test(b), 'no raw status or code is shown');
  check(await visible(page.locator('#ag-broken-exp')) && await visible(page.locator('#ag-broken-del')), 'the card offers Export raw and Delete this chat');
  check(await page.evaluate(() => document.querySelector('#ag-title').textContent) === 'broken one', 'the title bar names the chat');
  await snap('broken-opened');
  // a reload with the broken chat remembered (the page reopens the last chat on load): the same card, no error toast
  await page.evaluate(() => { try { localStorage.setItem('pxa-control.agent.session', JSON.stringify('c_corrupt0001')); } catch (e) { /* none */ } });
  await open(BASE + (BASE.includes('?') ? '&' : '?') + 'theme=' + THEME + '#chat');
  check(await until(() => !!document.querySelector('#ag-broken-card'), 10000), 'a reload that tries to reopen the broken chat shows the same card');
  check(await page.evaluate(() => { const t = document.querySelector('#toast'); return !t || t.hidden || !t.textContent; }), 'and no error toast');
  await snap('broken-after-reload');
  // Delete from the list, with the confirm
  await showList();
  await page.locator('.ag-item.broken[data-id="c_corrupt0001"] button.ag-broken-b').click();
  check(await until(() => !!document.querySelector('.ag-item.confirm .ag-del-yes'), 5000), 'Delete asks to confirm');
  await snap('delete-confirm');
  await page.click('.ag-item.confirm .ag-del-yes');
  check(await until(() => !document.querySelector('.ag-item.broken'), 8000), 'after the confirm the broken row is gone');
  const list = await page.evaluate(async () => (await (await fetch('/api/chat/sessions')).json()).sessions.map(s => s.id));
  check(!list.includes('c_corrupt0001'), 'the server no longer lists it');
  check(!(await page.locator('#ag-broken-card').count()), 'the card is gone; a fresh chat is shown');
  await snap('after-delete');
}

async function image() {
  console.log(`[${THEME} ${WIDTH}px] C14: the bad_image toast`);
  // make the SERVER refuse the image: the page sends a real PNG; the request is rewritten in flight to an unsupported type
  let rewrite = true;
  await page.route('**/api/chat/run', async route => {
    const req = route.request();
    if (!rewrite || req.method() !== 'POST') return route.continue();
    const j = JSON.parse(req.postData() || '{}');
    if (j.images && j.images.length) j.images[0].media_type = 'image/svg+xml';
    return route.continue({postData: JSON.stringify(j)});
  });
  await page.setInputFiles('#ag-attach', {name: 'pic.png', mimeType: 'image/png', buffer: Buffer.from(PNG, 'base64')});
  check(await until(() => document.querySelectorAll('#ag-files .ag-file').length === 1, 5000), 'the image chip is shown');
  await page.fill('#ag-in', 'what is in this picture?');
  await page.press('#ag-in', 'Enter');
  check(await until(() => { const t = document.querySelector('#toast'); return t && !t.hidden && /image 1 is image\/svg\+xml/.test(t.textContent); }, 10000), 'a toast appears with the server\'s reason');
  const toast = await page.evaluate(() => { const t = document.querySelector('#toast'); return {text: t.textContent, cls: t.className, shown: !t.hidden}; });
  console.log('       toast: ' + toast.text);
  check(/use a PNG, JPEG, GIF or WebP image/.test(toast.text) && /bad/.test(toast.cls), 'it is an error toast and says what to use instead');
  await snap('bad-image-toast');
  const b = await body();
  check(/Remove the image, then send again/.test(b), 'a notice in the chat repeats the reason and says what to do');
  check(await page.evaluate(() => document.querySelectorAll('.ag-u').length) === 0, 'no unsent message bubble is left behind');
  check(await page.evaluate(() => document.querySelector('#ag-in').value) === 'what is in this picture?', 'the typed words are back in the box');
  check(await page.evaluate(() => document.querySelectorAll('#ag-files .ag-file').length) === 1, 'the image chip is still there to remove');
  check(await page.evaluate(() => { const s = document.querySelector('#ag-send'); return s.getAttribute('aria-label') === 'Send' && !s.disabled; }), 'Send is usable again');
  await snap('bad-image-notice');
  await page.click('#ag-files .ag-file-x');
  check(await page.evaluate(() => document.querySelectorAll('#ag-files .ag-file').length) === 0, 'the bad image is removed');
  rewrite = false;
  await page.press('#ag-in', 'Enter');
  check(await idle(), 'sending again without it works and the reply finishes');
  check((await page.evaluate(() => document.querySelectorAll('.ag-a .ag-md').length)) >= 1, 'a reply is shown');
  await snap('sent-after');
}

(async () => {
  const browser = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await browser.newContext({viewport: {width: W, height: 900}, deviceScaleFactor: 1});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error' && !/ERR_|Failed to load resource|status of (400|401|404|422)|net::/.test(m.text())) errors.push('console: ' + m.text()); });
  await open(BASE + (BASE.includes('?') ? '&' : '?') + 'theme=' + THEME + '#chat');
  try {
    if (PHASE === 'broken') await broken();
    else if (PHASE === 'image') await image();
    else throw new Error('unknown phase ' + PHASE);
  } catch (e) { failures.push('script error: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')); console.log('  FAIL script error: ' + e.message); try { await snap('error'); } catch (x) { /* none */ } }
  check(errors.length === 0, 'the page logged no error' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
  await browser.close();
  console.log(failures.length ? 'FAILED: ' + failures.length : 'ALL OK');
  process.exit(failures.length ? 1 : 0);
})();
