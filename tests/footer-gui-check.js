// FR#15: the app footer (Discord, Ko-fi, the PXA version, the Update button) on every page of PXA Control, Home first.
// Run it with tests/footer-gui-check.py (hermetic env, a fake pxa-update with an update waiting):
//     node footer-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME
// Per page (every tab in the bar): one footer in the document, shown, with both links, the version and the Update button; scrolled to
// the bottom, the page's last element ends above the footer and the footer is in sight above the phone's bottom bar; in keyboard
// order the footer comes right after the page's last control; no sideways scroll. Then the footer's Update button opens the Update dialog from Home.
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME] = process.argv.slice(2);
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) {
  shot++;
  await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: false});
}
async function until(fn, ms, step) { const t0 = Date.now(); for (;;) { let v; try { v = await fn(); } catch (e) { v = null; } if (v) return v; if (Date.now() - t0 > ms) return v; await sleep(step || 250); } }
const SHOTS = ['home', 'models', 'servers'];

// what the page shows of the footer, measured after a scroll to the very bottom
function measure() {
  // checkVisibility: also false inside a closed <details> (its content is laid out but content-visibility hidden)
  const vis = e => !!e && e.getClientRects().length > 0 && e.checkVisibility({visibilityProperty: true});
  const foots = [...document.querySelectorAll('.foot')];
  const shown = foots.filter(vis);
  const f = shown[0] || null;
  const sec = [...document.querySelectorAll('main>section')].find(s => !s.hidden) || null;
  window.scrollTo(0, document.documentElement.scrollHeight);
  const r = {count: foots.length, shown: shown.length, page: sec && sec.dataset.page};
  if (!f) return r;
  const fr = f.getBoundingClientRect();
  r.links = [...f.querySelectorAll('a')].map(a => a.href);
  r.ver = (f.querySelector('#sv-ver') || {}).textContent || '';
  const up = f.querySelector('#sv-up');
  r.up = !!up && vis(up) ? up.textContent.trim() : '';
  r.top = Math.round(fr.top); r.bottom = Math.round(fr.bottom); r.vh = innerHeight;
  // the bottom tab bar on a phone is fixed: the footer must clear it
  const nav = document.getElementById('tabs');
  const np = nav && getComputedStyle(nav).position === 'fixed' ? nav.getBoundingClientRect().top : innerHeight;
  r.navTop = Math.round(np);
  // the page's lowest element that is laid out in the flow (not fixed, not clipped by a scroller inside the page)
  let low = sec ? sec.getBoundingClientRect().bottom : 0, lowEl = sec ? sec.id : '';
  if (sec) for (const e of sec.querySelectorAll('*')) {
    if (!vis(e)) continue;
    const cs = getComputedStyle(e);
    if (cs.position === 'fixed' || e.closest('dialog,[popover]')) continue;
    let clipped = false;
    for (let p = e.parentElement; p && p !== sec; p = p.parentElement) {
      const pc = getComputedStyle(p);
      if (pc.overflowY !== 'visible' || pc.overflowX !== 'visible') { clipped = true; break; }
    }
    if (clipped) continue;
    const b = e.getBoundingClientRect().bottom;
    if (b > low + 0.5) { low = b; lowEl = e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.className && typeof e.className === 'string' ? '.' + e.className.split(' ')[0] : ''); }
  }
  r.low = Math.round(low); r.lowEl = lowEl;
  r.after = !!sec && !!(sec.compareDocumentPosition(f) & Node.DOCUMENT_POSITION_FOLLOWING);
  r.sideways = document.documentElement.scrollWidth > document.documentElement.clientWidth + 1;
  return r;
}

// keyboard order: Shift+Tab from the footer's first link lands in the open page (its last control, whatever kind it is),
// and Tab from there comes back to the footer: the footer follows the page, it is not stuck somewhere else in the order
async function tabIntoFooter() {
  await page.evaluate(() => document.querySelector('.foot a').focus());
  await page.keyboard.press('Shift+Tab');
  const back = await page.evaluate(() => {
    const a = document.activeElement, sec = [...document.querySelectorAll('main>section')].find(s => !s.hidden);
    return {inPage: !!(a && sec && sec.contains(a)), text: a ? (a.tagName.toLowerCase() + (a.id ? '#' + a.id : '') + ' ' + (a.textContent || '').trim().slice(0, 30)) : ''};
  });
  await page.keyboard.press('Tab');
  const fwd = await page.evaluate(() => document.activeElement === document.querySelector('.foot a'));
  return {...back, fwd};
}

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: +WIDTH, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, THEME);
  const tag = `[${THEME} ${WIDTH}px]`;
  try {
    await page.goto(BASE + '/#home');
    await until(() => page.evaluate(() => !!(S.info && S.update && S.update.binary)), 30000);
    const tabs = await page.evaluate(() => [...document.querySelectorAll('#tabs button[role=tab]')].map(x => x.dataset.tab));
    check(tabs[0] === 'home' && tabs.length >= 12, `${tag} the tab bar: ${tabs.join(',')}`);
    for (const t of tabs) {
      console.log(`${tag} page ${t}`);
      await page.evaluate(x => showTab(x), t);
      await sleep(t === 'home' || t === 'servers' || t === 'models' ? 2500 : 1200);
      await page.mouse.move(2, 2);
      const m = await page.evaluate(measure);
      check(m.count === 1, `${t}: one footer in the document (${m.count})`);
      // the shot comes first: a page without the footer (the base) is pictured too, scrolled to its bottom
      if (SHOTS.includes(t)) { await page.evaluate(() => { document.activeElement && document.activeElement.blur(); window.scrollTo(0, document.documentElement.scrollHeight); }); await sleep(300); await snap(t); }
      check(m.shown === 1, `${t}: the footer is shown`);
      if (m.shown !== 1) continue;
      check(m.links.includes('https://discord.pxanetwork.com/') && m.links.includes('https://ko-fi.com/shatteredrealms1'), `${t}: the Discord and Ko-fi links`);
      check(/^PXA Control \S/.test(m.ver), `${t}: the version: "${m.ver}"`);
      check(m.up === 'Update', `${t}: the Update button (an update waits): "${m.up}"`);
      check(m.after, `${t}: the footer comes after the page in the document`);
      check(m.low <= m.top + 1, `${t}: the page's last element (${m.lowEl}, bottom ${m.low}) ends above the footer (top ${m.top})`);
      check(m.bottom <= m.navTop + 1 && m.top >= 0, `${t}: scrolled to the bottom, the footer is in sight (${m.top}-${m.bottom}) above the bottom bar (${m.navTop}) of ${m.vh}`);
      check(!m.sideways, `${t}: no sideways page scroll`);
      const k = await tabIntoFooter();
      check(k.inPage && k.fwd, `${t}: keyboard order: Shift+Tab from the footer lands on the page's last control ("${k.text}"), Tab comes back to the footer`);
    }
    console.log(`${tag} the footer's Update button on Home`);
    await page.evaluate(() => showTab('home'));
    await sleep(800);
    const upShown = await page.locator('#sv-up').isVisible();
    check(upShown, 'Home: the footer Update button is shown');
    if (upShown) {
      await page.locator('#sv-up').scrollIntoViewIfNeeded();
      await page.locator('#sv-up').click();
      const open = await until(() => page.evaluate(() => document.getElementById('up').open), 8000, 100);
      check(!!open, 'Home: the footer Update button opens the Update dialog');
      await until(async () => !/Checking/.test(await page.locator('#up-sub').innerText()), 15000, 100);
      check(/4\.0-beta\.6/.test(await page.locator('#up-sub').innerText()), 'the dialog names the waiting build');
      await page.click('#up-cancel');
    }
  } catch (e) {
    failures.push('exception: ' + e.message); console.log('  FAIL exception: ' + e.message);
  }
  check(errors.length === 0, `${tag} no page error` + (errors.length ? ': ' + errors[0] : ''));
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log(`all checks passed (${THEME} ${WIDTH}px)`);
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
