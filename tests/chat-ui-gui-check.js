// Headless-browser check of the chat Assistant page's hardening (todo control-chat-hardening, page side), run by
// tests/chat-ui-gui-check.py which starts the hermetic Control + FAKE llama-server (tools/pxa_control/tests) and this script:
//     node chat-ui-gui-check.js BASE_URL(with ?token=) WIDTH OUT_DIR THEME PHASE HELPER_URL [LONG_DOWN_S]
// PHASE (one per run):
//   plain    bidi controls (U06), a pasted text file over 100k characters (C15), a 1 MB reply (S21, with numbers), a dropped
//            stream that resumes with no repeated text (E07 part 1)
//   restart  Control killed mid-stream and restarted with the SAME token: "Reconnecting...", then the saved chat's interrupted turn (E07)
//   newtoken Control killed and restarted with a NEW token (what a restart does on 127.0.0.1): "signed out" note + sign-in (E07)
//   longdown Control down for LONG_DOWN_S seconds (more than the old 12 retries): still "Reconnecting...", no "Lost the connection", then back (E07)
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const fs = require('fs');
const [BASE, WIDTH, OUT, THEME, PHASE, HELPER, LONGDOWN] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, ctx, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${PHASE}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: false}); }
async function until(fn, ms = 30000, arg) { try { await page.waitForFunction(fn, arg, {timeout: ms, polling: 200}); return true; } catch (e) { return false; } }
const body = () => page.evaluate(() => document.body.innerText);
const idle = () => until(() => { const b = document.querySelector('#ag-send'); return b && b.getAttribute('aria-label') === 'Send' && !b.disabled; }, 60000);
const helper = async p => (await (await fetch(HELPER + p)).json());

async function open(url) {
  await page.addInitScript(() => {            // keep every EventSource so a test can fake a dropped connection
    window.__es = [];
    const E = window.EventSource;
    window.EventSource = function (u, o) { const e = new E(u, o); window.__es.push(e); return e; };
    window.EventSource.prototype = E.prototype;
    for (const k of ['CONNECTING', 'OPEN', 'CLOSED']) window.EventSource[k] = E[k];
  });
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForSelector('#ag-root #ag-in', {timeout: 60000});
  await until(() => { const b = document.querySelector('#ag-send'); return b && !b.disabled; }, 60000);
}
async function send(text) {
  await page.fill('#ag-in', text);
  await page.press('#ag-in', 'Enter');
}
const replyText = () => page.evaluate(() => Array.from(document.querySelectorAll('.ag-a .ag-md')).map(e => e.textContent).join('\n'));
const BIDI_RE = /[‪-‮⁦-⁩]/;

async function plain() {
  console.log(`[${THEME} ${WIDTH}px] U06: bidi controls are shown as markers, not obeyed`);
  await send('[[fake:unicode]]');
  check(await idle(), 'the reply finished');
  let b = await body();
  check(!BIDI_RE.test(b), 'no bidi control character is left in the page text');
  check(b.includes('[U+202E]') && b.includes('[U+202C]'), 'the override and its end are shown as visible [U+202E] [U+202C] markers');
  check(b.includes('שלום') && b.includes('你好世界'), 'ordinary right-to-left and CJK text is untouched');
  await snap('bidi-reply');
  await send('[[fake:call:read_file:{"path":"evil\\u202Etxt.exe"}]]');
  check(await idle(), 'the tool turn finished');
  await page.evaluate(() => document.querySelectorAll('.ag-tool').forEach(d => d.open = true));
  b = await body();
  check(!BIDI_RE.test(b), 'tool rows (file names) carry no bidi control character');
  check(/evil\[U\+202E\]txt\.exe/.test(b), 'the file name shows the override as a marker: evil[U+202E]txt.exe');
  await snap('bidi-tool-row');
  await page.evaluate(() => document.querySelector('#ag-new, [aria-label="New chat"]') && document.querySelector('#ag-new, [aria-label="New chat"]').click());

  console.log(`[${THEME} ${WIDTH}px] C15: a pasted text file over 100k characters says it was cut`);
  await page.setInputFiles('#ag-attach', {name: 'big-log.txt', mimeType: 'text/plain', buffer: Buffer.from('word '.repeat(30000))});   // 150,000 characters
  check(await until(() => /were cut/.test(document.querySelector('#toast').textContent) && !document.querySelector('#toast').hidden, 5000), 'a notice appears');
  const toast = await page.evaluate(() => document.querySelector('#toast').textContent);
  console.log('       toast: ' + toast);
  check(/150,000 characters/.test(toast) && /first 100,000/.test(toast) && /last 50,000/.test(toast), 'it names the file size and the cut size');
  const v = await page.evaluate(() => document.querySelector('#ag-in').value);
  check(v.length > 100000 && v.length < 100200 && /\[\.\.\. 50,000 more characters of big-log\.txt were not added\]$/.test(v), 'a marker stays in the box (length ' + v.length + ')');
  await snap('paste-cut-notice');
  await page.fill('#ag-in', '');
  await page.evaluate(() => { document.querySelector('#toast').textContent = ''; });
  await page.setInputFiles('#ag-attach', {name: 'ok.txt', mimeType: 'text/plain', buffer: Buffer.from('short file')});
  await sleep(500);
  check((await page.evaluate(() => document.querySelector('#ag-in').value)) === 'short file' && !/were cut/.test(await page.evaluate(() => document.querySelector('#toast').textContent)), 'a small file is added whole, with no notice');
  await page.fill('#ag-in', '');

  console.log(`[${THEME} ${WIDTH}px] E07: a dropped stream resumes with no repeated text`);
  await send('[[fake:slow:2]]');
  check(await idle(), 'the clean run finished');
  const clean = await replyText();
  console.log('       clean reply: ' + JSON.stringify(clean.slice(-80)));
  await send('[[fake:slow:2]]');
  check(await until(() => document.querySelectorAll('.ag-a .ag-md').length >= 2 && document.querySelectorAll('.ag-a .ag-md')[1].textContent.length > 0, 20000), 'the second reply started');
  await page.route('**/api/chat/events*', r => r.abort());
  await page.evaluate(() => window.__es.forEach(e => e.dispatchEvent(new Event('error'))));
  check(await until(() => /Reconnecting to PXA Control/.test(document.body.innerText), 15000), 'the page says "Reconnecting to PXA Control..." while the stream is cut');
  await snap('reconnecting');
  await sleep(3000);
  await page.unroute('**/api/chat/events*');
  check(await idle(), 'after the connection is back the answer finishes');
  const all = await replyText();
  check(all === clean + '\n' + clean, 'the resumed reply equals the undisturbed one: no lost and no repeated text');
  check(!/Reconnecting to PXA Control|Lost the connection/.test(await body()), 'no reconnect or give-up note is left');
  await snap('resumed');

  // S21: a 1 MB reply, two shapes, each delivered like a fast stream (about one delta per frame): no blank lines at all (one
  // giant paragraph: the worst case for a block renderer) and paragraphs (8 lines, a blank line)
  console.log(`[${THEME} ${WIDTH}px] S21: 1 MB replies`);
  await page.evaluate(() => {
    const m = window.__m = {gaps: [], long: 0, render: 0, calls: 0, chars: 0};
    let last = performance.now();
    (function tick() { const n = performance.now(); m.gaps.push(n - last); last = n; requestAnimationFrame(tick); })();
    try { new PerformanceObserver(l => l.getEntries().forEach(e => { m.long += e.duration; })).observe({entryTypes: ['longtask']}); } catch (e) { /* none */ }
    const r = pxaMd.render;
    pxaMd.render = function (s) { const t = performance.now(); const h = r.apply(this, arguments); m.render += performance.now() - t; m.calls++; m.chars += String(s).length; return h; };
  });
  for (const shape of ['', ':md']) {
    await page.evaluate(() => { const m = window.__m; m.gaps.length = 0; m.long = 0; m.render = 0; m.calls = 0; m.chars = 0; });
    const t0 = Date.now();
    await send('[[fake:big:1000000:2000:33' + shape + ']]');
    const ok = await idle();
    const wall = Date.now() - t0;
    check(ok, '1 MB reply (' + (shape ? 'paragraphs' : 'one paragraph') + ') finished (' + wall + ' ms)');
    const unitBase = 'BIG-' + 'x'.repeat(59) + '\n', unit = shape ? unitBase.repeat(8) + '\n' : unitBase;
    const want = (unit.repeat(Math.floor(1000000 / unit.length) + 1)).slice(0, 1000000).replace(/\s+/g, '');
    const res = await page.evaluate(() => {
      const m = window.__m, g = m.gaps.slice(5).sort((a, b) => a - b);
      const last = Array.from(document.querySelectorAll('.ag-a')).pop();
      const txt = Array.from(last.querySelectorAll('.ag-md')).map(e => e.textContent).join('').replace(/\s+/g, '');
      return {frame_gap_max_ms: Math.round(g[g.length - 1]), frame_gap_p95_ms: Math.round(g[Math.floor(g.length * .95)]), frames: g.length, frames_over_100ms: g.filter(x => x > 100).length,
              longtask_total_ms: Math.round(m.long), md_render_total_ms: Math.round(m.render), md_render_calls: m.calls, md_chars_rendered: m.chars, page_text_chars: txt.length,
              segments: document.querySelectorAll('.ag-seg').length, text: txt};
    });
    const got = res.text; delete res.text;
    res.stream_wall_ms = wall; res.shape = shape ? 'paragraphs' : 'one-paragraph';
    console.log('METRIC big ' + JSON.stringify(res));
    fs.appendFileSync(`${OUT}/${THEME}-${WIDTH}-big-metrics.jsonl`, JSON.stringify(res) + '\n');
    check(got === want, 'the page holds exactly the 1,000,000 characters sent, no loss, no repeat (' + got.length + ' vs ' + want.length + ' non-space characters)');
    await snap('big-reply' + (shape ? '-paragraphs' : ''));
    await page.evaluate(() => window.scrollTo(0, 0));
  }
}

async function newChatAndStall() {
  await send('[[fake:stall:300]]');
  check(await until(() => (document.querySelector('.ag-a .ag-md') || {textContent: ''}).textContent.length > 0, 30000), 'the stream started');
}
async function restart(newToken, longDown) {
  console.log(`[${THEME} ${WIDTH}px] E07: Control restarted mid-stream (${newToken ? 'new token' : 'same token'}${longDown ? ', down ' + longDown + ' s' : ''})`);
  await newChatAndStall();
  await helper('/crash');
  check(await until(() => /Reconnecting to PXA Control/.test(document.body.innerText), 40000), 'the page says "Reconnecting to PXA Control..."');
  await snap('reconnecting');
  const down = longDown || 4;
  const end = Date.now() + down * 1000;
  while (Date.now() < end) await sleep(1000);
  if (longDown) {
    const b = await body();
    check(/Reconnecting to PXA Control/.test(b) && !/Lost the connection/.test(b), 'after ' + down + ' s down (the old limit was ~60 s) it still reconnects and never gives up');
    await snap('still-reconnecting');
  }
  const st = await helper('/start?token=' + (newToken ? 'new' : 'same'));
  if (newToken) {
    check(await until(() => /signed out/i.test(document.body.innerText), 60000), 'the page says you are signed out (not "Lost the connection")');
    const b = await body();
    check(!/Lost the connection|Reconnecting to PXA Control/.test(b), 'no give-up and no reconnect spinner is left');
    check((await page.locator('.ag-signin-note input.ag-signin').count()) === 1 && (await page.locator('.ag-signin-note button', {hasText: 'Sign in again'}).count()) === 1, 'the sign-in action (address box + Sign in again) is there');
    await snap('signed-out');
    await page.fill('.ag-signin', st.url);
    await page.click('.ag-signin-note button:has-text("Sign in again")');
    await page.waitForSelector('#ag-root #ag-in', {timeout: 60000});
    check(await until(() => /Interrupted: PXA Control stopped/.test(document.body.innerText), 60000), 'after signing in the saved chat opens and shows the interrupted turn');
    await snap('signed-in-again');
  } else {
    check(await until(() => /Interrupted: PXA Control stopped/.test(document.body.innerText), 90000), 'once Control is back the saved chat is reopened with the interrupted turn');
    const b = await body();
    check(!/Lost the connection|Reconnecting to PXA Control/.test(b), 'no give-up and no reconnect spinner is left');
    check((await page.locator('button:has-text("Resend")').count()) === 1, 'a Resend button is offered');
    await snap('interrupted');
  }
}

(async () => {
  const browser = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  ctx = await browser.newContext({viewport: {width: W, height: 900}, deviceScaleFactor: 1});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error' && !/ERR_|Failed to load resource|status of (401|404)|net::/.test(m.text())) errors.push('console: ' + m.text()); });
  await open(BASE + (BASE.includes('?') ? '&' : '?') + 'theme=' + THEME + '#chat');
  try {
    if (PHASE === 'plain') await plain();
    else if (PHASE === 'restart') await restart(false, 0);
    else if (PHASE === 'newtoken') await restart(true, 0);
    else if (PHASE === 'longdown') await restart(false, +LONGDOWN || 70);
    else throw new Error('unknown phase ' + PHASE);
  } catch (e) { failures.push('script error: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')); console.log('  FAIL script error: ' + e.message); try { await snap('error'); } catch (x) { /* none */ } }
  check(errors.length === 0, 'the page logged no error' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
  await browser.close();
  console.log(failures.length ? 'FAILED: ' + failures.length : 'ALL OK');
  process.exit(failures.length ? 1 : 0);
})();
