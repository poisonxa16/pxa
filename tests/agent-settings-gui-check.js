// Headless-browser check of the chat Assistant view's settings panel, and of what that page asks for while it loads
// (tools/pxa_chat/ui/agent.js + agent.css). Run it with tests/agent-settings-gui-check.py, which starts the environment
// (tests/encode_env.py: a fake rig, a fake engine, PXA Control) and the browser:
//     node agent-settings-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR
// It covers the two things a UI pass of PXA Control found broken on 2026-10-09:
//   A. the settings gear and the drawer it opens were reachable only in Advanced mode (both carried .ag-advonly,
//      which [data-ag-mode=simple] hides with !important), so from the chat page's default Simple mode there was
//      no way to open the settings at all - and the host-access switch (#ag-host-cb) lives inside that drawer;
//   B. every visit to the chat tab asked GET /api/thinking?sid=<a stopped server> and the browser logged the 400
//      it came back with, twice on a load: a stopped target has no thinking profile to show.
// Exit code 0 only when every check passed and the page logged no error and asked nothing that came back 4xx.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT] = process.argv.slice(2);
const failures = [], errors = [], bad = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: false}); }
const visible = sel => page.locator(sel).first().isVisible();
const text = sel => page.locator(sel).first().innerText();
const widthOf = sel => page.locator(sel).first().evaluate(e => Math.round(e.getBoundingClientRect().width));
// wait for a condition without turning a slow answer into a crashed scenario
async function until(fn, ms = 20000) { try { await page.waitForFunction(fn, null, {timeout: ms}); return true; } catch (e) { return false; } }

(async () => {
  const browser = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await browser.newContext({viewport: {width: +WIDTH, height: 900}, deviceScaleFactor: 1});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  // every answer the page gets, so a 4xx cannot hide in a handler that swallows it
  page.on('response', r => { if (r.status() >= 400) bad.push(r.status() + ' ' + r.request().method() + ' ' + r.url()); });

  console.log(`[${WIDTH}px] the chat page loads`);
  await page.goto(BASE + '/#chat', {waitUntil: 'networkidle'});
  await page.waitForSelector('#ag-root .ag-comp #ag-in', {timeout: 30000});
  await sleep(1200);
  const mode = await page.evaluate(() => document.querySelector('#p-chat').dataset.agMode);
  check(mode === 'simple', 'the chat starts in Simple mode (data-ag-mode=' + mode + ')');
  check(bad.length === 0, 'nothing on the way in came back 4xx' + (bad.length ? ': ' + bad.join(' | ') : ''));

  console.log(`[${WIDTH}px] A. the settings are reachable in Simple mode`);
  check(await visible('#ag-gear'), 'the settings gear is on screen in Simple mode');
  check(!(await visible('#ag-drawer')), 'the settings drawer starts closed');
  await page.click('#ag-gear'); await sleep(500);
  check(await visible('#ag-drawer'), 'the gear opens the settings drawer');
  check(await widthOf('#ag-drawer') > 200, 'the open drawer has a real column (' + (await widthOf('#ag-drawer')) + 'px)');
  check(!(await visible('#ag-sys')), 'Simple mode hides the advanced knobs (system prompt)');
  await snap('chat-settings-simple');

  console.log(`[${WIDTH}px] A. the host-access switch can be worked from there`);
  check((await page.locator('#ag-host-cb').count()) === 1, 'the drawer holds the host-access switch (#ag-host-cb)');
  check(await visible('#ag-host-cb'), 'the host-access switch is on screen');
  check(!(await page.locator('#ag-host-cb').isDisabled()), 'the host-access switch is enabled on this Control (machine-local)');
  // wait on the panel's own words, not on the checkbox: the box flips natively on the click, the panel only
  // after the server has answered, so waiting on the box would read the old state back
  await page.click('#ag-host-cb');
  check(await until(() => /On for this chat/.test(document.querySelector('#ag-host-w').innerText)), 'turning it on is confirmed in the panel, not only in the checkbox');
  check(await page.locator('#ag-host-cb').isChecked(), 'the switch shows as on too');
  await page.click('#ag-host-cb');
  check(await until(() => /Off\./.test(document.querySelector('#ag-host-w').innerText)), 'turning it back off is confirmed too');
  const st = await (await fetch(BASE + '/api/chat/host')).json();
  check(st.available === true && st.on === false, 'the switch round-trips through the backend and ends off: ' + JSON.stringify(st));

  console.log(`[${WIDTH}px] A. Advanced still shows everything`);
  await page.click('#ag-mode button[data-v=advanced]'); await sleep(500);
  check(await visible('#ag-drawer'), 'switching to Advanced leaves the drawer open');
  check(await visible('#ag-sys'), 'Advanced shows the system prompt (the knobs Simple hides)');
  await snap('chat-settings-advanced');
  await page.click('#ag-mode button[data-v=simple]'); await sleep(300);
  // the drawer's own close button, which is what a phone has (there the open drawer is an overlay over the
  // composer, so the gear is behind it - that is the drawer's design, not a second way to be stuck)
  check(await visible('#ag-drawer-x'), 'the drawer carries its own close button');
  await page.click('#ag-drawer-x'); await sleep(400);
  check(!(await visible('#ag-drawer')), 'the drawer close button closes it');
  if (+WIDTH > 900) {
    await page.click('#ag-gear'); await sleep(400);
    check(await visible('#ag-drawer'), 'the gear reopens it');
    await page.click('#ag-gear'); await sleep(400);
    check(!(await visible('#ag-drawer')), 'and the gear closes it again');
  }

  console.log(`[${WIDTH}px] B. no tab and no reload asks anything that fails`);
  for (const t of ['servers', 'models', 'launch', 'rig', 'encode', 'live', 'chat']) {
    await page.click(`button[data-tab="${t}"]`); await sleep(1200);
  }
  await page.reload({waitUntil: 'networkidle'}); await sleep(1500);
  await page.click('button[data-tab="chat"]'); await sleep(1500);
  check(bad.length === 0, 'nothing asked during the pass came back 4xx' + (bad.length ? ': ' + bad.join(' | ') : ''));
  check(errors.length === 0, 'the page logged no error' + (errors.length ? ': ' + errors.join(' | ') : ''));
  // asked from here, not from the page: the page doing it would be the failure this section is checking for
  const think = (await fetch(BASE + '/api/thinking?sid=main')).status;
  check(think === 400, 'the endpoint itself still answers 400 for a stopped server (the page just must not ask): ' + think);

  await browser.close();
  console.log(failures.length ? 'RESULT: FAIL (' + failures.length + ')' : 'RESULT: PASS');
  process.exit(failures.length ? 1 : 0);
})().catch(async e => { console.log('SCENARIO CRASHED:', e.message); try { await snap('crash'); } catch (x) { /* */ } console.log('PAGE ERRORS:', errors.join(' | ')); process.exit(2); });
