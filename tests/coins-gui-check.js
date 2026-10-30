// Real-browser click-through of the PXA Coins panel (Store tab), driven against a LOCAL licd (tests/coins-gui-check.py starts it with a
// scratch db and throwaway keys, and PXA Control pointed at it).
//     node coins-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME HOOKS_URL LICD_URL KOFI_TOKEN TRIAGE_TOKEN
// The flow, in the order a person meets it:
//   no wallet -> create it -> balance 0 and the earn table licd serves (not the page's own list) -> a donation lands by itself (the poll,
//   N3) -> a confirmed report lands by itself -> "Get a report code" shows a code and an award made with it lands -> redeem a supporter
//   reward -> the saved statement follows every change (N3) -> over-balance (the dialog says how many coins are missing; a balance that
//   drops under an open dialog is refused with licd's words) -> double redeem (a redeem made elsewhere while the dialog is open: "already
//   unlocked, no coins used", one spend row) -> held wallet (a chargeback below zero: unlocks paused, the extras stay) -> a donation
//   brings it back -> store offline: the signed balance with the "Offline" note -> back online.
// Every state is checked for words (coins / supporter rewards only: never paid / sell / buy / price), no horizontal page scroll, no page error.
// Exit code 0 only when every check passed.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, HOOKS, LICD, KOFI, TRIAGE] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
const panel = async () => norm(await page.locator('#wl-root').innerText());
const BANNED = /\b(paid|pay|pays|sell|sold|selling|buy|bought|purchase|purchased|price|prices|priced|pricing)\b/i;
async function until(fn, what, ms) {
  const end = Date.now() + (ms || 20000);
  let last;
  while (Date.now() < end) { try { last = await fn(); if (last) return last; } catch (e) { /* not yet */ } await sleep(250); }
  failures.push('timeout: ' + what); console.log('  FAIL timeout: ' + what);
  return null;
}
const balance = () => page.locator('.wl-bal').evaluate(e => +e.firstChild.textContent.replace(/[^0-9-]/g, '')).catch(() => null);
const waitBalance = (n, ms) => until(async () => (await balance()) === n, 'the balance shows ' + n + ' (it is ' + '?)', ms || 25000);
const hook = async (path, body) => (await fetch(HOOKS + path, body ? {method: 'POST', body: JSON.stringify(body)} : {})).json();
async function kofi(txn, amount, wid) {
  const d = {verification_token: KOFI, kofi_transaction_id: txn, message_id: 'm-' + txn, type: 'Donation', amount, currency: 'USD',
             email: 'fan@example.org', from_name: 'Fan', message: 'thanks ' + wid};
  const r = await fetch(LICD + '/v1/hooks/kofi', {method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({data: JSON.stringify(d)}).toString()});
  return r.json();
}
async function award(body) {
  const r = await fetch(LICD + '/v1/svc/award', {method: 'POST', headers: {'Content-Type': 'application/json', Authorization: 'Bearer ' + TRIAGE}, body: JSON.stringify(body)});
  return [r.status, await r.json()];
}
async function noScroll(label) {
  const g = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(g.sw <= g.cw + 1, `no horizontal page scroll ${label} (${g.sw} <= ${g.cw})`);
}
async function words(label) {
  // the whole Store page as a person sees it (the panel, the catalogue cards below it, an open dialog, the toast)
  const t = norm(await page.evaluate(() => document.querySelector('#p-store').innerText + ' ' + (document.querySelector('#sd[open]') ? document.querySelector('#sd').innerText : '') + ' ' + document.querySelector('#toast').innerText));
  const m = t.match(BANNED);
  check(!m, `wording ${label}: coins / supporter rewards only${m ? ' (found "' + m[0] + '")' : ''}`);
}
const dlgText = async () => norm(await page.locator('#sd').innerText());
const toastText = async () => norm(await page.locator('#toast').innerText());
async function waitToast(re, what) {
  return until(async () => { const t = await toastText(); return re.test(t) && !(await page.locator('#toast').getAttribute('hidden') !== null) ? t : null; }, what, 8000);
}
async function closeDlg() { if (await page.locator('#sd[open]').count()) { await page.locator('#sd .acts button, #sd #sd-acts button').first().click(); await sleep(200); } }
const card = id => page.locator(`#wl-root .pcard[data-item=${id}]`);

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push('console: ' + m.text()); });
  await page.addInitScript(([t]) => {
    try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ }
    window.walletPollMs = 3000;            // the background refresh, shortened for the test (release: 60 s)
  }, [THEME]);
  await page.goto(BASE + '/#store');
  await page.waitForSelector('#wl-root .wl', {timeout: 30000});
  await sleep(800);
  const tag = `[${THEME} ${WIDTH}px]`;

  // ---- 1. no wallet yet
  console.log(`${tag} no wallet yet`);
  let t = await panel();
  check(/Create my wallet/.test(t), 'the panel offers "Create my wallet"');
  check(await page.getByRole('button', {name: 'Connect this computer'}).count() === 1 && await page.getByRole('button', {name: 'Recover my wallet'}).count() === 1, 'Connect and Recover are offered too');
  await words('before a wallet'); await noScroll('before a wallet'); await snap('no-wallet');

  // ---- 2. create
  console.log(`${tag} create, the balance, the earn table`);
  await page.getByRole('button', {name: 'Create my wallet'}).click();
  await waitBalance(0, 15000);
  t = await panel();
  const wid = (/PXW-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}/.exec(t) || [''])[0];
  check(!!wid, 'the wallet ID is shown: ' + wid);
  check(/No coins yet/.test(t), 'an empty history says so');
  const earn = norm(await page.locator('.wl-earn').innerText());
  check(/Donation\s*10 per 1 US dollar/.test(earn), 'the earn table is licd\'s ("Donation: 10 per 1 US dollar"): ' + earn.slice(0, 80));
  check(!/A one-off donation on Ko-fi/.test(earn), "the page's own built-in list is not what is shown");
  check(/Monthly earn limit: 150 coins/.test(t), 'the monthly earn limit comes from licd: 150 coins');
  check(await page.locator('.wl-earn li').count() >= 6, 'the earn table lists the ways to earn (' + await page.locator('.wl-earn li').count() + ' rows)');
  const cost = await page.locator('#wl-root .pcard[data-item] .b.sup').allInnerTexts();
  check(cost.length === 2 && cost.every(x => /Unlock for 100 coins/.test(x)), 'two supporter rewards, each "Unlock for 100 coins": ' + cost.map(norm).join(' | '));
  check(await page.getByRole('button', {name: 'Get a report code'}).isVisible(), 'the report-link button is visible');
  check(await page.getByRole('link', {name: /Donate on Ko-fi/}).isVisible(), 'the donate link is visible');
  await words('with an empty wallet'); await noScroll('with an empty wallet'); await snap('wallet-empty');
  let st = await hook('/statement');
  check(st.statement && st.statement.wallet === wid && st.statement.coins === 0, 'N3: Control already saved a signed statement for the new wallet (nobody pressed Save): ' + JSON.stringify(st.statement));

  // ---- 3. a donation lands without a reload (the poll)
  console.log(`${tag} a donation lands by itself`);
  const ko = await kofi('txn-gui-1', '10.00', wid);
  check(ko.result === 'credited' && ko.coins === 100, 'licd credited the Ko-fi donation: 100 coins');
  check(await waitBalance(100), 'N3: the balance follows the donation with no reload (100)');
  t = await panel();
  check(/From donations\s*100/.test(t) && /\+100/.test(t), 'the split and the history show the donation (+100)');
  st = await hook('/statement');
  check(st.statement && st.statement.coins === 100, 'N3: the saved statement followed the donation: coins ' + (st.statement && st.statement.coins));
  await snap('after-donation');

  // ---- 4. a confirmed report, the report code
  console.log(`${tag} earn: a confirmed report, the report code`);
  const [s1, a1] = await award({source: 'bug', class: 'correctness', bug_id: 'gui-1', wallet: wid});
  check(s1 === 200 && a1.granted === 30, 'licd credited a confirmed report (30 coins)');
  check(await waitBalance(130), 'N3: the balance follows the report (130)');
  check(/Earned\s*30/.test(await panel()), 'the split shows Earned 30');
  await page.getByRole('button', {name: 'Get a report code'}).click();
  await page.waitForSelector('#sd[open] .wl-id', {timeout: 10000});
  const dt = await dlgText();
  const code = (/PXR-[0-9A-Z]{4}-[0-9A-Z]{4}/.exec(dt) || [''])[0];
  check(!!code && /GitHub or Discord/.test(dt), 'the report code dialog shows a code and where to put it: ' + code);
  await words('the report code dialog'); await noScroll('the report code dialog'); await snap('report-code');
  const [s2, a2] = await award({source: 'bug', class: 'small', bug_id: 'gui-2', link_code: code});
  check(s2 === 200 && a2.granted === 10 && a2.wallet === wid, 'an award made with that code lands in this wallet (+10)');
  await closeDlg();
  check(await waitBalance(140), 'the balance follows (140)');
  st = await hook('/statement');
  check(st.statement && st.statement.coins === 140, 'the saved statement followed (140)');
  console.log(`${tag} Link Discord: the one-time /wallet link command`);
  check(await page.getByRole('button', {name: 'Link Discord'}).count() === 1, 'an unlinked wallet offers "Link Discord"');
  await page.getByRole('button', {name: 'Link Discord'}).click();
  await page.waitForSelector('#sd[open] .wl-id', {timeout: 10000});
  const ld = await dlgText();
  check(/\/wallet link:PXJ-[0-9A-Z]{4}-[0-9A-Z]{4}/.test(ld) && /10 minutes/.test(ld), 'the dialog shows "/wallet link:PXJ-...." and how long it works: ' + ((/\/wallet link:\S+/.exec(ld) || [''])[0]));
  await words('the Link Discord dialog'); await noScroll('the Link Discord dialog'); await snap('link-discord');
  await closeDlg();

  // ---- 5. over the balance: a balance that drops under an open dialog (licd refuses, in its words)
  console.log(`${tag} over the balance`);
  await card('anibal').locator('.b.sup').click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  let d = await dlgText();
  check(/Unlock Anibal for 100 coins\?/.test(d) && /Your balance after\s*40 coins/.test(d), 'the confirm dialog names the cost and the balance after (40): ' + d.slice(0, 90));
  await hook('/chargeback', {wallet: wid, coins: 100});                       // the balance falls to 40 while the dialog is open
  await sleep(4500);                                                           // a poll goes by: the open dialog is not redrawn
  check(await page.locator('#sd[open]').count() === 1, 'a background refresh does not close an open dialog');
  await page.locator('#sd .pri').click();
  const bad = await waitToast(/needs 100 coins and this wallet has 40/i, 'the refusal toast says what is missing');
  check(!!bad, 'over-balance is refused with a clear message: ' + bad);
  check(await page.locator('#sd[open]').count() === 1 && !(await page.locator('#sd .pri').isDisabled()), 'the dialog stays and its button works again');
  await snap('refused-over-balance');
  await closeDlg();
  check(await waitBalance(40), 'after the dialog closes the panel shows the real balance (40)');
  check(await hook('/spend-rows').then(r => r.n) === 0, 'nothing was spent');
  await card('pxqn').locator('.b.sup').click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  d = await dlgText();
  check(/Not enough coins yet/.test(d) && /60 coins more/.test(d) && /Ko-fi/.test(d), 'under the cost the dialog says how many coins are missing (60) and links Ko-fi: ' + d.slice(0, 120));
  await words('the not-enough dialog'); await snap('not-enough-dialog');
  await closeDlg();

  // ---- 6. redeem a supporter reward
  console.log(`${tag} redeem a supporter reward`);
  await kofi('txn-gui-2', '10.00', wid);
  check(await waitBalance(140), 'a second donation lands (140)');
  // one balance on the whole page: the nav badge and the catalogue cards below the panel read the wallet's coins, not an older snapshot
  await until(async () => norm(await page.locator('#nav-credits').innerText()) === '140', 'the nav badge shows the wallet balance (140)', 8000);
  const below = norm(await page.locator('#st-root').innerText());
  check(!/undefined|NaN/.test(await page.locator('#p-store').innerText()), 'no "undefined" or "NaN" anywhere on the Store page');
  check(!/Your coins/i.test(below) && !/Needs \d+ coins more/.test(below), 'the catalogue below does not show a second, different balance or "Needs ... more" while the wallet has 140');
  check(await page.locator('#st-root').getByRole('button', {name: /Unlock for 100 coins/}).count() === 2, 'the catalogue cards offer "Unlock for 100 coins" (wallet coins)');
  await page.locator('#st-root').getByRole('button', {name: /Unlock for 100 coins/}).first().click();   // the card goes through the same wallet dialog
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  check(/Unlock .* for 100 coins\?/.test(await dlgText()), 'a catalogue card opens the wallet\'s unlock dialog: ' + (await dlgText()).slice(0, 60));
  await closeDlg();
  await card('anibal').locator('.b.sup').click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  await page.locator('#sd .pri').click();
  await waitToast(/Anibal is unlocked/, 'the unlocked toast');
  check(await waitBalance(40), 'the balance after the unlock is 40');
  await until(async () => /Unlocked/.test(norm(await card('anibal').innerText())), 'the Anibal card says Unlocked');
  t = await panel();
  check(/Redeemed\s*100/.test(t) && /-100/.test(t), 'the split and the history show the redeem (-100)');
  check(await hook('/spend-rows').then(r => r.n) === 1, 'exactly one spend row on licd');
  st = await hook('/statement');
  check(st.statement && st.statement.coins === 40 && (st.statement.owned || []).includes('anibal'), 'N3: the saved statement followed the redeem at once (40, anibal owned)');
  await words('after a redeem'); await noScroll('after a redeem'); await snap('after-redeem');

  // ---- 7. double redeem: the same reward redeemed elsewhere while this dialog is open
  console.log(`${tag} double redeem`);
  await kofi('txn-gui-3', '10.00', wid);
  check(await waitBalance(140), 'a third donation lands (140)');
  await card('pxqn').locator('.b.sup').click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  const first = await page.evaluate(async () => (await api('/api/store/wallet/redeem', {method: 'POST', body: {item_id: 'pxqn'}, timeout: 60000})).redeemed);
  check(first === true, 'the reward is redeemed from elsewhere while the dialog is open');
  await page.locator('#sd .pri').click();
  const again = await waitToast(/already/i, 'the second redeem says it was already unlocked');
  check(!!again && /no coins were used/i.test(again), 'a double redeem is refused with a clear message: ' + again);
  await sleep(300);
  check(await hook('/spend-rows').then(r => r.n) === 2, 'the second redeem is not a second spend (2 spend rows: one per reward)');
  check(await waitBalance(40), 'the balance is unchanged by the second redeem (40)');
  await until(async () => /Unlocked/.test(norm(await card('pxqn').innerText())), 'the card says Unlocked');
  await snap('double-redeem');

  // ---- 7b. gift coins to another supporter: the form, the preview (nothing moves), the confirm, the history
  console.log(`${tag} gift coins`);
  const other = (await hook('/other-wallet', {})).wallet;
  await page.getByRole('button', {name: 'Gift coins'}).click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  await page.locator('#sd input').nth(0).fill(other);
  await page.locator('#sd input').nth(1).fill('15');
  await words('the gift form'); await noScroll('the gift form'); await snap('gift-form');
  await page.getByRole('button', {name: 'Review the gift'}).click();
  const pv = await until(async () => { const d = await dlgText(); return /Gift 15 coins to/.test(d) ? d : null; }, 'the gift preview dialog');
  check(!!pv && pv.includes(other) && /Your balance after\s*25 coins/.test(pv), 'the preview names the wallet and the balance after (25)');
  check(await hook('/gift-rows').then(r => r.n) === 0 && await balance() === 40, 'a preview moves nothing (0 gift rows, balance 40)');
  await words('the gift preview'); await noScroll('the gift preview'); await snap('gift-preview');
  await page.getByRole('button', {name: 'Send 15 coins'}).click();
  const sent = await waitToast(/Sent 15 coins/, 'the gift toast');
  check(!!sent && sent.includes(other), 'the toast says the gift was sent to that wallet');
  check(await waitBalance(25), 'the balance follows the gift (25)');
  t = await panel();
  check(/Gift to PXW-/.test(t) && /Gifts\s*-15/.test(t), 'the history and the split show the gift (-15)');
  check(await hook('/gift-rows').then(r => r.n) === 2, 'one gift = two ledger rows (out and in)');
  await words('after a gift'); await snap('after-gift');
  await kofi('txn-gui-3b', '1.50', wid);
  check(await waitBalance(40), 'a small donation brings it back to 40');

  // ---- 8. held wallet: a chargeback below zero
  console.log(`${tag} held wallet`);
  await hook('/chargeback', {wallet: wid, coins: 100});
  check(await waitBalance(-60), 'the balance follows the chargeback (-60)');
  t = await panel();
  check(/Unlocks are paused/.test(t) && /Extras you already unlocked keep working/.test(t), 'a held wallet says unlocks are paused and unlocked extras keep working');
  check(/Reversed\s*200/.test(t), 'the split shows Reversed 200 (two chargebacks of 100)');
  check(await card('anibal').locator('.chip.ok').count() === 1 && await card('pxqn').locator('.chip.ok').count() === 1, 'both rewards still read Unlocked');
  await words('a held wallet'); await noScroll('a held wallet'); await snap('held');
  await kofi('txn-gui-4', '10.00', wid);
  check(await waitBalance(40), 'a donation brings the balance back above zero by itself (40)');
  check(!/Unlocks are paused/.test(await panel()), 'the held note goes away');
  st = await hook('/statement');
  check(st.statement && st.statement.coins === 40, 'N3: the saved statement followed the hold and its release (40)');

  // ---- 9. store offline: the signed balance, then back
  console.log(`${tag} store offline`);
  await hook('/offline', {on: true});
  await until(async () => /Offline/.test(await panel()), 'the panel says Offline', 15000);
  t = await panel();
  check(balance && await balance() === 40, 'offline the signed balance still shows (40)');
  check(/as of/.test(t) && /signed by PXA/.test(t), 'the offline note says "as of ... signed by PXA"');
  check(await page.getByRole('button', {name: 'Get a report code'}).isDisabled(), 'actions that need a connection are disabled offline');
  await words('offline'); await noScroll('offline'); await snap('offline');
  await hook('/offline', {on: false});
  await until(async () => !/Offline/.test(await panel()), 'the panel is live again', 15000);
  check(await balance() === 40, 'back online: the live balance (40)');

  // ---- 10. redeem a voucher (a code from the voucher era or one PXA sent): the coins land in this wallet
  console.log(`${tag} redeem a voucher`);
  const vc = (await hook('/voucher', {coins: 25})).code;
  await page.getByRole('button', {name: 'Redeem a voucher'}).click();
  await page.waitForSelector('#sd[open] input', {timeout: 5000});
  await page.locator('#sd input').fill('pxd1.not-a-real-code');
  await page.locator('#sd .pri').click();
  const vbad = await waitToast(/voucher/i, 'a wrong voucher code is refused');
  check(!!vbad && await balance() === 40, 'a wrong voucher code is refused and nothing changes (40): ' + String(vbad).slice(0, 80));
  if (!(await page.locator('#sd[open]').count())) { await page.getByRole('button', {name: 'Redeem a voucher'}).click(); await page.waitForSelector('#sd[open] input', {timeout: 5000}); }
  await page.locator('#sd input').fill(vc);
  await words('the voucher dialog'); await noScroll('the voucher dialog'); await snap('voucher');
  await page.locator('#sd .pri').click();
  await waitToast(/Voucher redeemed/, 'the voucher toast');
  check(await waitBalance(65), 'the voucher\'s 25 coins land in the wallet (65)');
  check(/Voucher redeemed/.test(await panel()), 'the history says "Voucher redeemed"');

  console.log(`${tag} errors`);
  check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.join(' || ').slice(0, 300) : ''));
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log(`all checks passed (${THEME} ${WIDTH}px)`);
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
