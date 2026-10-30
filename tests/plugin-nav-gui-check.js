// A plugin's own top-level tab in PXA Control (tests/plugin-nav-gui-check.py starts Control with the real Wayhouse plugin, allowed, off).
//     node plugin-nav-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME
// The tab button arrives at boot before Plugins -> the tab says "Wayhouse is off" with Turn on -> the page runs in a sealed frame that
// fills the tab -> a project and a chat answer through the host -> a deep link opens the tab -> Manage in Plugins goes back -> off.
// Exit code 0 only when every check passed.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { await sleep(200); shot++; await page.screenshot({path: `${OUT}/nav-${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`}); }
const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
const BANNED = /\b(paid|pay|pays|sell|sold|selling|buy|bought|purchase|purchased|price|prices|priced|pricing)\b/i;
async function until(fn, what, ms) {
  const end = Date.now() + (ms || 20000);
  while (Date.now() < end) { try { const v = await fn(); if (v) return v; } catch (e) { /* not yet */ } await sleep(250); }
  failures.push('timeout: ' + what); console.log('  FAIL timeout: ' + what);
  return null;
}
async function noScroll(label) {
  const g = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(g.sw <= g.cw + 1, `no horizontal page scroll ${label} (${g.sw} <= ${g.cw})`);
}
const btn = () => page.locator('#tabs button[data-tab="plug-wayhouse"]');
const sec = () => page.locator('#p-plug-wayhouse');
const shown = async sel => page.locator(sel).first().isVisible().catch(() => false);

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 860}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push(String(e)));
  await page.addInitScript(([t]) => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, [THEME]);
  const tag = `[${THEME} ${WIDTH}px]`;
  try {
    console.log(`${tag} the tab button`);
    await page.goto(BASE + '/');
    await until(async () => (await btn().count()) === 1, 'the Wayhouse button in the tab bar');
    check(norm(await btn().innerText()) === 'Wayhouse', 'the button says Wayhouse');
    check(await btn().locator('svg').count() >= 1, 'the button has its glyph');
    const order = await page.evaluate(() => [...document.querySelectorAll('#tabs button[role=tab]')].map(x => x.dataset.tab));
    check(order.indexOf('plug-wayhouse') === order.indexOf('plugins') - 1, 'it sits right before Plugins (' + order.join(',') + ')');
    check(order.filter(t => t.startsWith('plug-')).length === 1, 'one plugin tab (the plugin asked for one)');
    await noScroll('with the extra tab');
    await snap('tabbar');

    console.log(`${tag} off, then on`);
    await btn().click();
    await until(() => sec().isVisible(), 'the Wayhouse section opens');
    check(await btn().getAttribute('aria-selected') === 'true', 'the button is the selected tab');
    check(!(await shown('#p-plugins')), 'the Plugins section is not shown');
    await until(async () => /Wayhouse is off/.test(norm(await sec().innerText())), 'the off panel says Wayhouse is off');
    check(await sec().locator('[data-act=start]').count() === 1, 'it offers Turn on Wayhouse');
    await noScroll('on the off panel');
    await snap('off');
    await sec().locator('[data-act=start]').click();
    await until(async () => (await sec().locator('iframe.ptframe').count()) === 1, 'the page frame after Turn on', 30000);
    const f = sec().locator('iframe.ptframe');
    check(await f.getAttribute('sandbox') === 'allow-scripts', 'the frame is sealed (sandbox allow-scripts only)');
    check(await f.getAttribute('data-plugin') === 'wayhouse', 'the frame belongs to wayhouse');
    const box = await f.boundingBox();
    const mw = await sec().evaluate(e => { const c = getComputedStyle(e); return Math.round(e.clientWidth - parseFloat(c.paddingLeft) - parseFloat(c.paddingRight)); });
    check(box && box.width >= mw - 4, `the frame fills the tab's width (${box && Math.round(box.width)} of ${mw})`);
    check(box && box.height >= 420, `the frame is tall (${box && Math.round(box.height)} px)`);
    check(/sealed frame/.test(norm(await sec().locator('.ptbar').innerText())), 'the bar says the page runs in a sealed frame');
    const pf = await page.evaluate(() => { const s = document.getElementById('p-plug-wayhouse'), f = document.querySelector('main > .foot');
      return {n: document.querySelectorAll('.foot').length, before: !!(s && f && (s.compareDocumentPosition(f) & Node.DOCUMENT_POSITION_FOLLOWING)),
              last: !!f && f.parentElement.lastElementChild === f, shown: !!f && f.checkVisibility()}; });
    check(pf.n === 1 && pf.before && pf.last && pf.shown, `the plugin's page sits above the app footer, which still closes the page (${JSON.stringify(pf)})`);

    console.log(`${tag} the page works through the host`);
    const fr = page.frameLocator('#p-plug-wayhouse iframe.ptframe');
    await until(() => fr.locator('#welcome').isVisible(), 'the Wayhouse welcome card', 30000);
    check(await fr.locator('html').evaluate(e => e.dataset.theme) === THEME, 'the page took Control\'s theme');
    check(/\.gguf \(card 0\)/.test(norm(await fr.locator('#models').innerText())), 'the running server is listed by its model file and card');
    await snap('page');
    await fr.locator('#welcome-new').click();
    await until(() => fr.locator('#np-name').isVisible(), 'the new-project form');
    await fr.locator('#np-name').fill('Rig notes');
    await fr.locator('#np-create').click();
    await until(() => fr.locator('#pane-chat').isVisible(), 'the project opens on Chat');
    await fr.locator('#composer').fill('How much memory does the P100 have?');
    await fr.locator('#send').click();
    await until(async () => (await fr.locator('.msg.assistant').count()) === 1, 'the answer from the local server', 30000);
    check(/16 GB/.test(norm(await fr.locator('.msg.assistant .body').innerText())), 'the answer is shown');
    await noScroll('with the page open');
    await snap('chat');

    console.log(`${tag} deep link, back to Plugins, off`);
    await page.goto(BASE + '/#plug-wayhouse');
    await page.reload();
    await until(async () => (await sec().isVisible()) && (await sec().locator('iframe.ptframe').count()) === 1, 'the deep link opens the Wayhouse tab with its page', 30000);
    const fr2 = page.frameLocator('#p-plug-wayhouse iframe.ptframe');
    await until(async () => { for (const s of ['#projects', '#tabs', '#welcome']) if (await fr2.locator(s).isVisible()) return true; return false; }, 'the page is back', 30000);
    if (W >= 760) check(/Rig notes/.test(norm(await fr2.locator('#projects').innerText())), 'the project is still there after the reload');
    await sec().locator('[data-act=manage]').click();
    await until(() => page.locator('#p-plugins').isVisible(), 'Manage in Plugins opens the Plugins tab');
    const card = page.locator('#ph-root .pcard[data-plugin="wayhouse"]');
    await until(async () => (await card.count()) === 1, 'the Wayhouse card in Plugins');
    const ct = norm(await card.innerText());
    check(/Wayhouse/.test(ct), 'the card names Wayhouse');
    const m = (ct + ' ' + norm(await sec().innerText())).match(BANNED);
    check(!m, 'wording: coins / supporter rewards only' + (m ? ' (found "' + m[0] + '")' : ''));
    await snap('plugins-card');
    const stop = card.locator('[data-act=stop]');
    if (await stop.count()) {
      await stop.click();
      await btn().click();
      await until(async () => /Wayhouse is off/.test(norm(await sec().innerText())), 'turned off: the tab shows the off panel again', 20000);
      check(await sec().locator('iframe').count() === 0, 'no frame once it is off');
      await snap('off-again');
    } else check(false, 'the card offers Turn off');
  } catch (e) {
    failures.push('exception: ' + e.message); console.log('  FAIL exception: ' + e.message);
  }
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
  await b.close();
  console.log(failures.length ? `${tag} FAILED ${failures.length}: ${failures.join('; ')}` : `${tag} ALL OK`);
  process.exit(failures.length ? 1 : 0);
})();
