// Headless-browser check of PXA Control's Live tab against the fake workload of tests/live_env.py.
// Run it with tests/live-gui-check.py, which starts the environment and the browser:
//     node live-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME] = process.argv.slice(2);
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
const count = sel => page.locator(sel).count();
async function css(sel, prop, idx) { return page.locator(sel).nth(idx || 0).evaluate((e, p) => getComputedStyle(e)[p], prop); }
async function hoverAt(sel, fx, idx) {          // the pointer over a mark, scrolled into view first (the page re-draws every 2 s: take the box fresh)
  const loc = page.locator(sel).nth(idx || 0);
  await loc.scrollIntoViewIfNeeded();
  const b = await loc.boundingBox();
  await page.mouse.move(b.x + b.width * fx, b.y + b.height * 0.5);
  await sleep(300);
}
(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: +WIDTH, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.addInitScript(t => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); localStorage.removeItem('pxa-control.lv.range'); } catch (e) { /* */ } }, THEME);
  await page.goto(BASE + '/#live');
  await page.waitForSelector('#lv-tiles .lv-tile', {timeout: 30000});
  await page.waitForSelector('#lh-dec svg', {timeout: 30000});
  await sleep(4000);

  console.log(`[${THEME} ${WIDTH}px] structure`);
  check(await page.locator('nav.tabs button[data-tab=live][aria-selected=true]').count() === 1, 'the Live tab is selected');
  check(await count('#lv-tiles .lv-tile') === 2, 'one tile per running server (2)');
  const names = (await page.locator('#lv-tiles .lv-name').allInnerTexts()).join('|');
  check(/Qwen3\.8-27B/.test(names) && /Flash-Next/.test(names), 'tiles are named after the servers: ' + names);
  check(await count('#lv-cards .lv-strip') === 4, 'one strip per card (4)');
  check(await count('#lv-cards .lv-panel svg') === 16, 'four mini charts per card strip');
  check(await count('#lv-host .lv-strip') === 1 && await count('#lv-host .lv-panel svg') === 3, 'a strip for this machine: memory, CPU, swap');
  check(await count('#lv-tiles .lv-meter') >= 1, 'a tile shows the KV cache fill as a meter');
  for (const id of ['dec', 'pre', 'acc', 'xhit']) check(await page.locator('#lc-' + id).isVisible(), `chart "${id}" is shown (the fake servers use speculation and an expert cache)`);
  check(await count('#lh-dec path.lv-line') === 2, 'decode chart: one thin line per server');
  check(await count('#lh-dec .lv-legend span') === 2, 'two series: a legend is present');
  check(await count('#lh-acc .lv-legend') === 0 && await count('#lh-xhit .lv-legend') === 0, 'one series: no legend box');
  const s1 = await css('#lh-dec path.lv-line', 'stroke', 0), s2 = await css('#lh-dec path.lv-line', 'stroke', 1);
  check(s1 !== s2 && /rgb\(57, 135, 229\)/.test(s1) && /rgb\(217, 89, 38\)/.test(s2), `lines wear the series colours by server (${s1} / ${s2})`);
  check((await css('#lh-dec path.lv-line', 'strokeWidth')) === '2px', 'lines are 2 px');
  const dashed = await page.evaluate(() => [...document.querySelectorAll('#p-live svg line, #p-live svg path')].filter(e => { const d = getComputedStyle(e).strokeDasharray; return d && d !== 'none'; }).length);
  check(dashed === 0, 'nothing in the Live charts is dashed');
  check(await count('#lh-dec .lv-tick') >= 6, 'axes carry tick labels');
  const txt = await page.locator('#p-live').innerText();
  check(!/\bnull\b|undefined|NaN/.test(txt), 'no null / undefined / NaN anywhere on the tab');
  const dec = (await page.locator('#lv-tiles .lv-tile').first().locator('.lv-big .v').first().innerText()).replace(/\s+/g, ' ');
  check(/^\d+(\.\d)?\s?t\/s$/.test(dec) || /^-\s?t\/s$/.test(dec), 'a tile shows decode speed in text: ' + dec);
  const tileNum = await css('#lv-tiles .lv-big .v', 'color'), page_text = await css('body', 'color');
  check(tileNum === page_text, 'headline numbers wear the text colour, not a series colour');
  const w = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(w.sw <= w.cw + 1, `no horizontal page scroll (${w.sw} <= ${w.cw})`);
  const api = await page.evaluate(async () => (await fetch('/api/live')).text());
  check(!/SECRET-PROMPT-TEXT/.test(api) && !/SECRET-PROMPT-TEXT/.test(await page.content()), 'a request prompt (carried by the server /slots) never reaches the page');

  // history: wait until the workload produced requests
  await page.waitForFunction(() => document.querySelectorAll('#lv-tl rect.lv-dec').length >= 2, null, {timeout: 120000}).catch(() => {});
  console.log('hover');
  await snap('overview');
  await hoverAt('#lh-dec svg .lv-hit', 0.93);
  check(await page.locator('#lh-dec .lv-tip').isVisible(), 'hovering the decode chart shows one tooltip');
  const tip = (await page.locator('#lh-dec .lv-tip').innerText()).replace(/\s+/g, ' ');
  check(/t\/s/.test(tip) && /\d\d:\d\d:\d\d/.test(tip), 'the tooltip lists the time and a t/s value: ' + tip);
  check(await count('#lh-dec .lv-cross') === 1 && await page.locator('#lh-dec .lv-cross').getAttribute('visibility') === 'visible', 'a crosshair follows the pointer');
  await snap('hover-decode');
  await page.mouse.move(2, 2); await sleep(300);
  check(!(await page.locator('#lh-dec .lv-tip').isVisible()), 'the tooltip goes away when the pointer leaves');
  const bars = await count('#lv-tl rect.lv-dec');
  check(bars >= 2, `the request timeline has bars (${bars})`);
  check(/\d+ requests? in this range .* tokens written at [\d.]+ t\/s/.test(await page.locator('#lv-tl-sum').innerText()), 'the timeline carries a one-line summary of the range');
  if (bars) {
    await hoverAt('#lv-tl rect.lv-dec', 0.5, Math.max(0, bars - 3));
    const t2 = (await page.locator('#lv-tl .lv-tip').innerText()).replace(/\s+/g, ' ');
    check(/prompt/.test(t2) && /wrote/.test(t2), 'a bar tooltip says prompt, prefill and tokens written: ' + t2.slice(0, 120));
    await snap('hover-timeline');
    await page.mouse.move(2, 2); await sleep(200);
  }
  await hoverAt('#lv-cards .lv-mini svg .lv-hit', 0.97, 1);
  check(await page.locator('#lv-cards .lv-mini .lv-tip:not([hidden])').count() === 1, 'a card strip chart shows a tooltip too');
  await page.mouse.move(2, 2); await sleep(200);

  console.log('range');
  const ticks = () => page.evaluate(() => [...document.querySelectorAll('#lh-dec .lv-tick')].map(e => e.textContent));       // SVG text has no innerText
  const before = await ticks();
  await page.click('#lv-range button[data-v="300"]'); await sleep(600);
  const after = await ticks();
  check(JSON.stringify(before) !== JSON.stringify(after), '5 min range changes the time axis (' + before.join(' ') + ' -> ' + after.join(' ') + ')');
  check(await page.locator('#lv-range button[aria-pressed=true]').getAttribute('data-v') === '300', 'the range control shows the selection');
  await snap('range-5min');
  await page.click('#lv-range button[data-v="3600"]'); await sleep(600);
  await snap('range-1h');
  await page.click('#lv-range button[data-v="900"]'); await sleep(400);

  console.log('theme');
  const tickBefore = await css('#lh-dec .lv-tick', 'fill');
  await page.click('#theme'); await sleep(500);
  const tickAfter = await css('#lh-dec .lv-tick', 'fill');
  check(tickBefore !== tickAfter, `light and dark change the chart chrome (${tickBefore} -> ${tickAfter})`);
  const s1b = await css('#lh-dec path.lv-line', 'stroke', 0);
  check(s1b === s1, 'a server keeps its colour across themes');
  await snap('other-theme');
  await page.click('#theme'); await sleep(400);

  console.log('servers tab');
  await page.click('nav.tabs button[data-tab=servers]'); await sleep(1500);
  const bc = await page.evaluate(() => [...document.querySelectorAll('#sv-list .srv')].map(c => [c.querySelector('h3 span:nth-child(2)').textContent, getComputedStyle(c).borderTopColor]));
  check(bc.length === 2 && bc[0][1] === s1 && bc[1][1] === s2, 'the Servers tab paints each server in the same colour as the Live tab: ' + JSON.stringify(bc));
  await page.click('nav.tabs button[data-tab=live]'); await sleep(1500);
  check(await count('#lv-tiles .lv-tile') === 2, 'coming back to the tab keeps the tiles');

  check(errors.length === 0, 'no page error or console error' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await b.close();
  console.log(failures.length ? `FAILED ${failures.length}: ` + failures.join('; ') : 'ALL PASSED');
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('crash: ' + e.stack); process.exit(2); });
