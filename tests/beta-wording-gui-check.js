// Real-browser check of the beta wording, driven by tests/beta-wording-gui-check.py against a LOCAL licd with the
// Valued-only beta on and a beta release out.
//     node beta-wording-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME
// 1. no wallet: the Beta card says the beta is a Valued Supporter reward (no coins) and shows the v3.1 / beta 1 handover line;
// 2. a wallet that is not Valued: Store > Wallet says "Beta access: for Valued Supporters (no coins)" + the handover line, and the Beta
//    card says the channel is for Valued Supporters (never "No beta is out right now", never a coin price for the beta).
// Every state: coins / supporter rewards only (never paid / sell / buy / price), no "your key is not one", no horizontal scroll, no page error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/beta-${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
const BANNED = /\b(paid|pay|pays|sell|sold|selling|buy|bought|purchase|purchased|price|prices|priced|pricing)\b/i;
const HANDOVER = /v3\.1 \/ beta 1 users: run \/beta in Discord\./;
async function until(fn, what, ms) {
  const end = Date.now() + (ms || 20000);
  let last;
  while (Date.now() < end) { try { last = await fn(); if (last) return last; } catch (e) { /* not yet */ } await sleep(250); }
  failures.push('timeout: ' + what); console.log('  FAIL timeout: ' + what);
  return null;
}
async function noScroll(label) {
  const g = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(g.sw <= g.cw + 1, `no horizontal page scroll ${label} (${g.sw} <= ${g.cw})`);
}
function words(t, label) {
  const m = t.match(BANNED);
  check(!m, `wording ${label}: coins / supporter rewards only${m ? ' (found "' + m[0] + '")' : ''}`);
  check(!/your key is not one/i.test(t), `wording ${label}: no "your key is not one"`);
  check(!/\/key\b/.test(t), `wording ${label}: points to /beta, not /key`);
}
const cardText = async () => norm(await page.locator('#up-bcard').innerText());
async function openCard(re) {        // the card loads in the background: wait for the state the step expects, then report it as it is
  await page.evaluate(() => openUpdate());
  await page.waitForSelector('#up[open]', {timeout: 10000});
  return (await until(async () => { const t = await cardText(); return re.test(t) ? t : null; }, 'the Beta card fills in', 20000)) || cardText();
}
async function closeCard() { await page.locator('#up-cancel').click(); await sleep(300); }

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push('console: ' + m.text()); });
  await page.addInitScript(([t]) => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, [THEME]);
  await page.goto(BASE + '/#store');
  await page.waitForSelector('#wl-root .wl', {timeout: 30000});
  await sleep(800);
  const tag = `[${THEME} ${WIDTH}px]`;

  console.log(`${tag} 1. no wallet: the Beta card`);
  let t = await openCard(/Valued Supporter reward/);
  check(t && /Valued Supporter reward and spends no coins\. Run \/beta in the PXA Discord/.test(t), 'no wallet: the beta is a Valued Supporter reward, no coins' + (t && /reward and spends/.test(t) ? '' : ' [card: ' + t + ']'));
  check(t && HANDOVER.test(t), 'Beta card shows the v3.1 / beta 1 handover line');
  check(await page.locator('#bc-v31').isVisible(), '#bc-v31 is visible');
  words(t || '', 'Beta card, no wallet');
  await noScroll('Beta card, no wallet');
  await snap('card-no-wallet');
  await closeCard();

  console.log(`${tag} 2. a wallet that is not Valued`);
  await page.locator('#wl-root button', {hasText: 'Create my wallet'}).click();
  t = await until(async () => { const x = norm(await page.locator('#wl-root').innerText()); return /Beta access:/.test(x) ? x : null; }, 'the wallet panel', 25000);
  check(t && /Beta access: for Valued Supporters \(no coins\)/.test(t), 'Store > Wallet: Beta access: for Valued Supporters (no coins)');
  check(t && HANDOVER.test(t), 'Store > Wallet shows the v3.1 / beta 1 handover line');
  check(!(await page.locator('#wl-root button', {hasText: 'Activate beta access'}).count()), 'no coin unlock button for the beta');
  words(norm(await page.locator('#p-store').innerText()), 'Store page, wallet not Valued');
  await noScroll('Store page');
  await page.locator('#wl-root .wl').first().screenshot({path: `${OUT}/beta-${THEME}-${WIDTH}-${String(++shot).padStart(2, '0')}-wallet-panel.png`});
  await page.evaluate(() => { if (typeof loadBeta === 'function') loadBeta(); });
  t = await openCard(/channel is for Valued Supporters/);
  check(t && /The beta channel is for Valued Supporters\. If you are one, run \/beta in the PXA Discord/.test(t), 'wallet not Valued: the card says Valued Supporters + run /beta');
  check(t && !/No beta is out right now/.test(t), 'wallet not Valued: not "No beta is out right now"');
  check(t && HANDOVER.test(t), 'Beta card keeps the handover line');
  check(!(await page.locator('#bc-go').isVisible()), 'no install / redeem button for a wallet that is not Valued');
  words(t || '', 'Beta card, wallet not Valued');
  await noScroll('Beta card, wallet not Valued');
  await snap('card-not-valued');
  await closeCard();

  for (const e of errors) check(false, e);
  await b.close();
  console.log(`${tag} ${failures.length ? 'FAILED ' + failures.length : 'all checks passed'}`);
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('  FAIL crash: ' + (e.stack || e)); process.exit(1); });
