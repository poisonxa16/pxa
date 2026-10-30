// Headless-browser check of the rig picker's offline screen (tools/pxa_control_ui/hosts.html), the screen the Android app
// shows when a rig does not answer (?offline=1&from=<rig>) and the PWA's service worker shows for this Control.
// Run it with tests/hosts-offline-gui-check.py, which serves the page and opens the "rig" when this script asks:
//     node hosts-offline-gui-check.js PICKER_BASE RIG_ORIGIN OUT_DIR
// Checks: no offline banner on a plain open; an unknown ?from= is dropped (never named, never opened by itself); a remembered
// rig is named (its remembered name) and the banner offers
// Try again; it counts down and keeps trying by itself; once the rig answers it opens the rig without a click.
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const fs = require('fs');
const [BASE, RIG, OUT] = process.argv.slice(2);
const failures = [], errors = [];
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
(async () => {
  const browser = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await browser.newContext({viewport: {width: 390, height: 844}});
  const page = await ctx.newPage();
  page.on('pageerror', e => errors.push(String(e)));
  // 1. plain open: no banner
  await page.goto(BASE + '/hosts.html');
  check(await page.locator('#state').isHidden(), 'plain open: no offline banner');
  await page.screenshot({path: OUT + '/1-plain.png', fullPage: true});
  // 2. offline, unknown rig (security review P10 L1): a ?from= that is not a remembered rig is dropped: the banner does not
  //    name it, nothing counts down, and the page never opens it by itself (no open redirect from a link)
  const q = '/hosts.html?offline=1&from=' + encodeURIComponent(RIG);
  await page.goto(BASE + q);
  const st = (await page.locator('#state').innerText()).replace(/\s+/g, ' ');
  const host = RIG.replace(/^https?:\/\//, '');
  check(await page.locator('#state').isVisible(), 'offline: banner shown');
  check(!st.includes(host) && st.includes('not answering'), 'offline, unknown rig: not named [' + st.slice(0, 120) + ']');
  check(await page.locator('#again').isVisible(), 'offline: Try again button');
  await sleep(4500);
  const r1 = await page.locator('#retry').innerText();
  check(!/Trying again by itself|Checking/.test(r1), 'offline, unknown rig: no probe, no countdown [' + r1 + ']');
  check(page.url() === BASE + q, 'offline, unknown rig: the page stays (' + page.url() + ')');
  await page.screenshot({path: OUT + '/2-offline-unknown.png', fullPage: true});
  // 3. a remembered rig is named by its name; the first probe fails (rig closed) and backs off
  await page.evaluate(r => localStorage.setItem('pxa.hosts', JSON.stringify([{url: r, name: 'Den rig', last: 1}])), RIG);
  await page.goto(BASE + q);
  const st2 = (await page.locator('#state').innerText()).replace(/\s+/g, ' ');
  check(st2.includes('Den rig (' + host + ') is not answering'), 'offline: remembered rig named [' + st2.slice(0, 120) + ']');
  await sleep(4500);
  const r2 = await page.locator('#retry').innerText();
  check(/in (\d+) s/.test(r2) && +r2.match(/in (\d+) s/)[1] > 3, 'offline: after a failed probe it backs off [' + r2 + ']');
  check(page.url().startsWith(BASE), 'offline: stays on the picker while the rig is down');
  await page.screenshot({path: OUT + '/3-offline-named.png', fullPage: true});
  // 4. the rig comes back: the page opens it by itself
  fs.writeFileSync(OUT + '/OPEN-RIG', '1');
  let ok = false;
  for (let i = 0; i < 45 && !ok; i++) { await sleep(1000); ok = page.url().startsWith(RIG + '/'); }
  check(ok, 'rig back: opened by itself (' + page.url() + ')');
  await page.screenshot({path: OUT + '/4-reopened.png', fullPage: true});
  check(errors.length === 0, 'no page errors ' + JSON.stringify(errors).slice(0, 300));
  await browser.close();
  console.log(failures.length ? 'FAILED ' + failures.length : 'ALL OK');
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('FAIL exception ' + e); process.exit(2); });
