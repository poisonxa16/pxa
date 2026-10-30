// Headless-browser check of the header row that names every running server (PXA Control, the "Servers" issue:
// "All running servers displayed in the header. Currently only the last one started is displayed.").
// Run it with tests/header-gui-check.py, which starts the environment (three fake servers) and the browser:
//     node header-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME [stop]
// The row has three states and this checks the two a width can be asked for:
//   wide    -- one chip per running server: model short name + :port + the state in words
//   narrow  -- one summary chip ("3 servers") whose dropdown lists them all
//   cramped -- no row at all (the header has no room left); never asserted here, only tolerated
// "stop" checks the header while servers stop instead, and stops every server of the environment on the way (the owner,
// 10-17: "if there was two running and now only one it should show just the one not one and a no server ... it should
// always reflect whats running and if nothing then say no servers").
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, PHASE] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
async function head(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, clip: {x: 0, y: 0, width: W, height: W <= 820 ? 60 : 70}}); }
const count = sel => page.locator(sel).count();
const txt = async sel => (await page.locator(sel).allInnerTexts()).join(' | ').replace(/\s+/g, ' ').trim();
async function css(sel, prop, idx) { return page.locator(sel).nth(idx || 0).evaluate((e, p) => getComputedStyle(e)[p], prop); }

// the chips of the row itself; the dropdown's rows are not chips of the header
const CHIP = '#fchips .fchip:not(.fmore)';

// what the header says, as a reader sees it: the pill's words (null when the pill is not shown) and how many running
// servers the row shows (its chips, or the number on the summary chip)
const header = () => page.evaluate(() => {
  const shown = e => !!e && e.getClientRects().length > 0 && getComputedStyle(e).visibility !== 'hidden';
  const box = document.querySelector('#fchips'), sum = box.querySelector('.fmore .fname');
  return {pill: shown(document.querySelector('#health')) ? document.querySelector('#htext').textContent.trim() : null,
          chips: !shown(box) ? 0 : sum ? +((/^(\d+) server/.exec(sum.textContent) || [0, -1])[1]) : box.querySelectorAll('.fchip:not(.fmore)').length};
});
// the row follows the fleet (read every 5 s), the pill the page's own server (every 3 s): wait for the row, then a poll more
async function settle(n) {
  for (let i = 0; i < 40 && (await header()).chips !== n; i++) await sleep(500);
  await sleep(3500);
  return header();
}
// The page is pointed at Rig 1, the environment's first server. Rig 1 and Rig 2 stop the way the Stop button stops them,
// then Rig 3: one server left, the page's own among the stopped; then none.
async function stopFlow() {
  console.log(`[${THEME} ${WIDTH}px] the header while servers stop`);
  const sid = Object.fromEntries(await page.evaluate(async () => (await api('/api/servers')).servers.map(s => [s.name, s.sid])));
  const stop = n => page.evaluate(s => api('/api/stop', {body: {sid: s}, timeout: 60000}), sid[n]);
  let h = await header();
  check(/Rig 1/.test(h.pill || ''), `while the page's server runs, the pill names it (pill: ${JSON.stringify(h.pill)})`);
  await stop('Rig 1'); await stop('Rig 2');
  h = await settle(1);
  check(h.chips === 1, `two of three stopped: the row shows the one still running (${h.chips})`);
  check(!/no server/i.test(h.pill || ''), `a server still runs, so the header does not say "no server" (pill: ${JSON.stringify(h.pill)})`);
  check(h.pill === null || /Rig 3/.test(h.pill), `anything the pill says is about the server that runs (pill: ${JSON.stringify(h.pill)})`);
  await head('header-one-left');
  await stop('Rig 3');
  h = await settle(0);
  check(h.chips === 0, `all stopped: no chip is left (${h.chips})`);
  check(/^no servers$/i.test(h.pill || ''), `nothing runs, so the header says "no servers" (pill: ${JSON.stringify(h.pill)})`);
  await head('header-none');
}
async function done(b) {
  console.log(`[${THEME} ${WIDTH}px] errors`);
  check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors[0] : ''));
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log(`all checks passed (${THEME} ${WIDTH}px${PHASE ? ' ' + PHASE : ''})`);
}

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, THEME);
  await page.goto(BASE + '/#servers');
  await page.waitForSelector('#sv-list .card.srv', {timeout: 30000});
  await sleep(3000);

  const running = await page.evaluate(() => document.querySelectorAll('#sv-list .card.srv').length);
  console.log(`[${THEME} ${WIDTH}px] the header names every running server (${running} running)`);
  check(running === 3, `the environment has three running servers (${running})`);
  check(await count('#fchips') === 1, 'the header carries a running-server row');
  check(await page.evaluate(() => {                    // it sits with the header's other status, not off in the page
    const box = document.querySelector('#fchips'), hd = document.querySelector('header.hd');
    if (!box || !hd || !hd.contains(box)) return false;
    return !!(box.compareDocumentPosition(document.querySelector('#health')) & Node.DOCUMENT_POSITION_FOLLOWING);
  }), 'the row is in the header, ahead of the status pill');
  if (PHASE === 'stop') { await stopFlow(); return done(b); }

  const summary = await count('#fchips .fmore') === 1;
  const chips = await count(CHIP);
  // 2026-10 revamp: the sections moved to a sidebar and the utilities into one Help menu, so a mid-size window may now
  // have room for every chip. Wide: one chip per server. Phone: the summary chip. In between: whichever the measurement
  // picked, but never a partial row.
  if (W >= 2000) check(!summary && chips === 3, `a window this wide shows one chip per server (${summary ? 'summary' : chips + ' chips'})`);
  else if (W <= 820) check(summary && chips === 0, `a phone shows the one summary chip (${summary ? 'summary' : chips + ' chips'})`);
  else check((summary && chips === 0) || (!summary && chips === 3), `a ${W}px window shows every chip or the one summary chip, never part of the row (${summary ? 'summary' : chips + ' chips'})`);

  if (summary) {
    console.log(`[${THEME} ${WIDTH}px] the summary chip`);
    const label = await txt('#fchips .fmore .fname');
    check(/^3 servers$/.test(label), `the summary chip counts them in words: "${label}"`);
    check(await css('#fchips .fmore .fcaret', 'visibility') !== 'hidden', 'the summary chip carries a caret (it opens a list)');
    check(await count('#fchips .fmenu') === 0, 'the list is shut until it is asked for');
    await head('header-summary');
    const cardTop = await page.locator('#sv-list .card.srv').first().evaluate(e => Math.round(e.getBoundingClientRect().top));
    await page.locator('#fchips .fmore').click();
    await page.waitForSelector('#fchips .fmenu', {timeout: 5000});
    check(await count('#fchips .fmenu .fchip') === 3, 'the dropdown lists every running server (3 rows)');
    const rows = await txt('#fchips .fmenu');
    check(/Qwen3\.8-27B/.test(rows) && /Flash-Next-PXQ4/.test(rows) && /gemma-4-12b-it/.test(rows), 'the rows name the model each server runs: ' + rows);
    check((rows.match(/:\d{2,5}/g) || []).length === 3, 'every row carries its port: ' + rows);
    const mpos = await css('#fchips .fmenu', 'position');
    const moved = await page.locator('#sv-list .card.srv').first().evaluate((e, t) => Math.round(e.getBoundingClientRect().top) !== t, cardTop);
    check(mpos === 'absolute' || mpos === 'fixed', `the list is out of the page's flow (${mpos})`);
    check(!moved, 'opening the list does not push the page around');
    await head('header-dropdown');
  } else {
    console.log(`[${THEME} ${WIDTH}px] the chips`);
    const t = await txt(CHIP);
    check(/Qwen3\.8-27B/.test(t) && /Flash-Next-PXQ4/.test(t) && /gemma-4-12b-it/.test(t), 'one chip per server, naming the model it runs: ' + t);
    check((t.match(/:\d{2,5}/g) || []).length === 3, 'every chip carries its port: ' + t);
    for (let i = 0; i < 3; i++) {
      const st = (await txt(`${CHIP} >> nth=${i}`)).match(/(serving|busy|loading|starting|stopped|exited|error|warn|ok)\b/i);
      check(!!st, `chip ${i + 1} says its state in words: "${await txt(`${CHIP} >> nth=${i}`)}"`);
      const left = await page.locator(CHIP).nth(i).evaluate(e => getComputedStyle(e).borderLeftColor);
      check(!/0px solid|rgb\(0, 0, 0\)/.test(left), `chip ${i + 1} wears its server's colour on the left edge (${left})`);
    }
    await head('header-chips');
  }

  // the row never paints over the header's own controls, and never widens the page
  const geo = await page.evaluate(() => {
    const box = document.querySelector('#fchips'), hd = document.querySelector('header.hd');
    const b = box.getBoundingClientRect(), h = hd.getBoundingClientRect();
    return {l: b.left, r: b.right, hl: h.left, hr: h.right, sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth};
  });
  check(geo.l >= geo.hl - 1 && geo.r <= geo.hr + 1, `the row stays inside the header (${Math.round(geo.l)}..${Math.round(geo.r)} in ${Math.round(geo.hl)}..${Math.round(geo.hr)})`);
  check(geo.sw <= geo.cw + 1, `no horizontal page scroll (${geo.sw} <= ${geo.cw})`);

  // a chip (or a dropdown row) lands on that server's card, marked, not on a guess
  console.log(`[${THEME} ${WIDTH}px] the jump`);
  const ROW = summary ? '#fchips .fmenu .fchip' : CHIP;   // the second row of either list: one specific server
  const show = async () => {
    if (!summary) return;
    if (await count('#fchips .fmenu') === 0) await page.locator('#fchips .fmore').click();
    await page.waitForSelector(ROW, {timeout: 5000});
  };
  await show();
  const said = (await txt(ROW + ' >> nth=1')).replace(/\s+/g, ' ');
  const port = said.match(/:(\d{2,5})/);
  check(!!port, 'the row we click carries a port: ' + said);
  await page.locator('nav.tabs button[data-tab=live]').click();      // stand somewhere else first
  await sleep(600);
  await show();
  await page.locator(ROW).nth(1).click();
  await page.waitForSelector('#sv-list .card.flash', {timeout: 32000});   // the card is drawn after /api/fleet answers (the page gives that 30 s)
  check(await count('nav.tabs button[data-tab=servers][aria-selected=true]') === 1, 'a click opens the Servers tab');
  const flashed = (await txt('#sv-list .card.flash')).replace(/\s+/g, ' ');
  check(flashed.includes(':' + port[1]), `the marked card is the one that chip named (port ${port[1]}): ${flashed.slice(0, 70)}`);
  check(await count('#fchips .fmenu') === 0, 'the dropdown closes behind the click');
  await snap('card-marked');
  await sleep(2200);
  check(await count('#sv-list .card.flash') === 0, 'the mark fades on its own');

  await done(b);
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
