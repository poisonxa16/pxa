// Real-browser click-through of the Plugins tab (S3), driven against a LOCAL licd and three signed stub plugins
// (tests/plugin-ui-gui-check.py starts them).     node plugin-ui-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR THEME HOOKS_URL
// The flow, in the order a person meets it:
//   the list (three plugins, each "Signed by PXA", nothing allowed yet) -> MISSING: the supporter plugin is Locked and offers "Unlock with 100
//   coins" (no wallet: it sends you to make one; 60 coins: "needs 40 more", a dialog with Ko-fi; 160 coins: the real unlock) -> ACTIVE: the unlock
//   spends the coins, the grant arrives by sync, the card says Unlocked -> the consent card (plain words for every permission, tick boxes, the
//   count in the button, "Not now" changes nothing) -> Allow: the plugin runs -> its page opens in a sealed frame (opaque origin: no cookie, no
//   storage, no parent, no direct fetch; the bridge reaches its own routes only; the theme arrives) -> Turn off -> REVOKED: the owner revokes,
//   the next sync switches it off and the card says "No longer unlocked" with the one honest next step.
// Every state is checked for words (coins / supporter rewards only: never paid / sell / buy / price), no horizontal page scroll, no page error.
// Exit code 0 only when every check passed.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, THEME, HOOKS] = process.argv.slice(2);
const W = +WIDTH;
const failures = [], errors = [];
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
async function snap(name) { await page.evaluate(() => { window.scrollTo(0, 0); if (document.activeElement && document.activeElement.blur) document.activeElement.blur(); }); await sleep(150); shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
const BANNED = /\b(paid|pay|pays|sell|sold|selling|buy|bought|purchase|purchased|price|prices|priced|pricing)\b/i;
async function until(fn, what, ms) {
  const end = Date.now() + (ms || 20000);
  while (Date.now() < end) { try { const v = await fn(); if (v) return v; } catch (e) { /* not yet */ } await sleep(250); }
  failures.push('timeout: ' + what); console.log('  FAIL timeout: ' + what);
  return null;
}
const hook = async (path, body) => (await fetch(HOOKS + path, body ? {method: 'POST', body: JSON.stringify(body)} : {})).json();
async function noScroll(label) {
  const g = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(g.sw <= g.cw + 1, `no horizontal page scroll ${label} (${g.sw} <= ${g.cw})`);
}
async function words(label) {
  const t = norm(await page.evaluate(() => document.querySelector('#p-plugins').innerText + ' ' + (document.querySelector('#sd[open]') ? document.querySelector('#sd').innerText : '') + ' ' + document.querySelector('#toast').innerText));
  const m = t.match(BANNED);
  check(!m, `wording ${label}: coins / supporter rewards only${m ? ' (found "' + m[0] + '")' : ''}`);
}
const dlgText = async () => norm(await page.locator('#sd').innerText());
const card = id => page.locator(`#ph-root .pcard[data-plugin="${id}"]`);
const cardText = async id => norm(await card(id).innerText());
async function closeDlg(label) {
  if (await page.locator('#sd[open]').count()) { await page.locator('#sd #sd-acts button, #sd #sd-acts a').first().click(); await sleep(250); }
}
async function reloadList() { await page.goto(BASE + '/#plugins'); await page.reload(); await page.waitForSelector('#ph-root .pcard', {timeout: 30000}); await sleep(500); }
async function toastText() { return norm(await page.locator('#toast').innerText()); }
async function waitToast(re, what) { return until(async () => { const t = await toastText(); return re.test(t) ? t : null; }, what, 10000); }
const api = (path, body) => page.evaluate(async ([p, b]) => {
  const r = await fetch(p, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b || {})});
  let j = null; try { j = await r.json(); } catch (e) { /* */ }
  return {status: r.status, body: j};
}, [path, body]);

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await b.newContext({viewport: {width: W, height: 900}, colorScheme: THEME});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  const blocked = [];                      // the sealed frame's own probe of Control is SUPPOSED to be refused by its CSP: count it, do not fail on it
  page.on('console', m => { if (m.type() !== 'error' || /Failed to load resource/.test(m.text())) return;
    if (/connect-src 'none'|Refused to connect/.test(m.text()) && /\/api\/info/.test(m.text())) blocked.push(m.text()); else errors.push('console: ' + m.text()); });
  await page.addInitScript(([t]) => { try { localStorage.setItem('pxa-control.theme', JSON.stringify(t)); } catch (e) { /* */ } }, [THEME]);
  await page.goto(BASE + '/#plugins');
  try { await page.waitForSelector('#ph-root .pcard', {timeout: 30000}); }
  catch (e) { await snap('NO-CARDS'); console.log('  errors: ' + errors.join(' || ')); console.log('  page: ' + norm(await page.locator('#p-plugins').innerText()).slice(0, 400)); throw e; }
  await sleep(800);
  const tag = `[${THEME} ${WIDTH}px]`;

  // ---- 1. the list
  console.log(`${tag} the list`);
  const ids = await page.locator('#ph-root .pcard').evaluateAll(es => es.map(e => e.dataset.plugin).sort());
  check(JSON.stringify(ids) === JSON.stringify(['dictate-stub', 'quiet-demo', 'voice-demo']), 'three plugins are listed: ' + ids);
  for (const id of ids) {
    const t = await cardText(id);
    check(/Signed by PXA/.test(t) && /made by PXA/.test(t) && /v0\.1\.0/.test(t), `${id}: name, version, signature: ` + t.slice(0, 90));
    check(/Not asked yet/.test(t), `${id}: nothing is allowed before you say so`);
  }
  check(/Unlocks with coins/i.test(await cardText('dictate-stub')) && !/Unlocks with coins/i.test(await cardText('voice-demo')), 'only the supporter plugin says "Unlocks with coins"');
  check(/Free/.test(await cardText('voice-demo')), 'the free plugin says Free');
  await words('the list'); await noScroll('the list'); await snap('list-missing');

  // ---- 1b. speech on this computer: the speech plugins' engines, below the list (never clicked here: it would download them)
  console.log(`${tag} speech`);
  const sp = page.locator('#ph-root section.phspeech[data-ph=speech]');
  check(await sp.count() === 1, 'one "Speech on this computer" panel, outside the plugin cards');
  const st = await sp.getAttribute('data-state'), spt = norm(await sp.innerText());
  check(/Speech on this computer/.test(spt) && /Used by .*Voice demo/.test(spt) && !/Used by .*Quiet/.test(spt), 'it names the plugins that need speech: ' + spt.slice(0, 120));
  check(['setup', 'ready', 'unsupported'].includes(st), 'it shows one of its states: ' + st);
  if (st === 'setup') {
    check(/one-time download of about \d+ MB/.test(spt), 'it says how much the setup downloads: ' + spt.slice(0, 220));
    check(norm(await sp.locator('[data-act=speech-setup]').innerText()) === 'Set up speech', 'the main button reads "Set up speech"');
    check(/^With other languages \(about \d+ MB\)$/.test(norm(await sp.locator('[data-act=speech-languages]').innerText())), 'the second button says what the other languages add');
  }
  const order = await page.evaluate(() => { const g = document.querySelector('#ph-root .pgrid'), s = document.querySelector('#ph-root .phspeech');
    return !!(g && s && (g.compareDocumentPosition(s) & Node.DOCUMENT_POSITION_FOLLOWING)); });
  check(order, 'the panel sits after the plugin cards');
  await words('the speech panel'); await noScroll('the speech panel');
  await snap('speech-panel');

  // ---- 2. MISSING: locked, no wallet yet
  console.log(`${tag} missing`);
  let t = await cardText('dictate-stub');
  check(await card('dictate-stub').getAttribute('data-gate') === 'missing', 'the supporter plugin is "missing" at the gate');
  check(/Locked/.test(t), 'its card says Locked: ' + t.slice(0, 160));
  const ub = card('dictate-stub').locator('[data-act=unlock]');
  check(norm(await ub.innerText()) === 'Unlock with 100 coins', 'the button reads "Unlock with 100 coins": ' + norm(await ub.innerText()));
  check(await card('dictate-stub').locator('[data-act=consent], [data-act=start]').count() === 0, 'a locked plugin offers no Allow or Turn on');
  await ub.click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  let d = await dlgText();
  check(/Make your coins wallet first/.test(d) && /Open the Store/.test(d), 'with no wallet the dialog sends you to make one: ' + d.slice(0, 110));
  await words('the no-wallet dialog'); await noScroll('the no-wallet dialog'); await snap('dialog-no-wallet');
  await closeDlg();

  // ---- 3. a wallet with too few coins, then enough
  console.log(`${tag} coins`);
  const mk = await api('/api/store/wallet/create', {});
  check(mk.status === 200 && mk.body && mk.body.wallet_id, 'a wallet was created (as the Store tab does): ' + (mk.body && mk.body.wallet_id));
  const wid = mk.body && mk.body.wallet_id;
  await hook('/deposit', {wallet: wid, coins: 60});
  await sleep(1300);                                           // the statement gap of this test
  await reloadList();
  const sb = card('dictate-stub').locator('[data-act=unlock]');
  check(/Needs 40 coins more/.test(norm(await sb.innerText())), 'with 60 coins the button says "Needs 40 coins more": ' + norm(await sb.innerText()));
  await sb.click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  d = await dlgText();
  check(/Not enough coins yet/.test(d) && /needs 100 coins; you have 60 coins/.test(d) && /Ko-fi/.test(d), 'the dialog says what is missing and links Ko-fi: ' + d.slice(0, 120));
  await words('the not-enough dialog'); await snap('dialog-not-enough');
  await closeDlg();
  await hook('/deposit', {wallet: wid, coins: 100});
  await sleep(1300);
  await reloadList();
  check(norm(await card('dictate-stub').locator('[data-act=unlock]').innerText()) === 'Unlock with 100 coins', 'with 160 coins the button reads "Unlock with 100 coins"');
  await snap('missing-enough-coins');

  // ---- 4. ACTIVE: the unlock spends the coins, the gate opens
  console.log(`${tag} active`);
  await card('dictate-stub').locator('[data-act=unlock]').click();
  await page.waitForSelector('#sd[open]', {timeout: 5000});
  d = await dlgText();
  check(/Unlock Dictate \(test stub\) with 100 coins\?/.test(d) && /Your balance after\s*60 coins/.test(d), 'the confirm dialog names the cost and the balance after (60): ' + d.slice(0, 120));
  await words('the unlock dialog'); await noScroll('the unlock dialog'); await snap('dialog-unlock');
  await page.locator('#sd .pri').click();
  check(!!await waitToast(/is unlocked/, 'the unlock toast'), 'the unlock went through: ' + await toastText());
  check(!!await until(async () => (await card('dictate-stub').getAttribute('data-gate')) === 'active', 'the card is active at the gate', 25000), 'the grant arrived by sync: the gate says active');
  t = await cardText('dictate-stub');
  check(/Unlocked on this computer/.test(t), 'the card says "Unlocked on this computer": ' + t.slice(0, 160));
  check(await card('dictate-stub').locator('[data-act=unlock]').count() === 0 && await card('dictate-stub').locator('[data-act=consent]').count() === 1, 'the unlock button gave way to "Review and allow"');
  await words('after the unlock'); await noScroll('after the unlock'); await snap('active-needs-consent');

  // ---- 5. the consent card (supporter plugin)
  console.log(`${tag} consent card`);
  await card('dictate-stub').locator('[data-act=consent]').click();
  await page.waitForSelector('#sd[open] .consent', {timeout: 5000});
  d = await dlgText();
  check(/Allow Dictate \(test stub\)\?/.test(d) && /is made by PXA/.test(d), 'the card names the plugin and who made it');
  const rows = await page.locator('#sd .consent li').evaluateAll(es => es.map(e => e.innerText.replace(/\s+/g, ' ').trim()));
  check(rows.some(r => /use your microphone/i.test(r)) && rows.some(r => /type text into the window you are using/i.test(r)) && rows.some(r => /turn speech into text on this computer/i.test(r)),
    'every permission is in plain words (microphone, typing, speech to text): ' + rows.join(' | '));
  const ticks = page.locator('#sd .consent input[type=checkbox]');
  check(await ticks.count() === 3 && (await ticks.evaluateAll(es => es.every(e => e.checked))), 'three tick boxes, all ticked to start');
  check(/Allow 3 of 3 and turn on/.test(norm(await page.locator('#sd .pri').innerText())), 'the button counts: "Allow 3 of 3 and turn on"');
  await ticks.nth(1).uncheck();
  check(/Allow 2 of 3 and turn on/.test(norm(await page.locator('#sd .pri').innerText())), 'unticking one changes the count: 2 of 3');
  const drows = await page.locator('#sd .consent li').evaluateAll(es => es.map(e => ({t: e.innerText.replace(/\s+/g, ' ').trim(), reach: e.dataset.reach})));
  const dr = re => drows.find(r => re.test(r.t));
  check(dr(/microphone/i).reach === 'sense' && /uses a sensor/.test(dr(/microphone/i).t), 'the microphone row is tagged "uses a sensor"');
  check(dr(/type text into the window/i).reach === 'out' && /reaches beyond PXA/.test(dr(/type text into the window/i).t), 'typing into another window is tagged "reaches beyond PXA"');
  check(dr(/turn speech into text/i).reach === 'local' && /stays on this computer/.test(dr(/turn speech into text/i).t) && !/reaches beyond PXA/.test(dr(/turn speech into text/i).t),
    'local speech to text is tagged "stays on this computer", never "reaches beyond PXA": ' + dr(/turn speech into text/i).t);
  // the modal covers the WHOLE viewport and the page behind is inert: scroll to the footer bar and try to reach its button
  await page.evaluate(() => { document.querySelector('#pl-root .alert a.b').scrollIntoView({block: 'center'}); });
  await sleep(200);
  const hit = await page.evaluate(() => { const a = document.querySelector('#pl-root .alert a.b'), r = a.getBoundingClientRect();
    const h = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2); return {onButton: !!h && (h === a || a.contains(h)), inDialog: !!h && !!h.closest('#sd'), tag: h && h.tagName,
      backdrop: getComputedStyle(document.querySelector('#sd'), '::backdrop').position, bg: getComputedStyle(document.querySelector('#sd'), '::backdrop').backgroundColor}; });
  check(!hit.onButton, 'with a dialog open the footer button below the fold cannot be hit (page behind is inert): hit ' + hit.tag);
  check(hit.backdrop === 'fixed' && hit.bg !== 'rgba(0, 0, 0, 0)', 'the backdrop is fixed over the viewport and dims: ' + hit.backdrop + ' ' + hit.bg);
  let clicked = true; try { await page.locator('#pl-root .alert a.b').click({timeout: 1500, trial: true}); } catch (e) { clicked = false; }
  check(!clicked, 'a click on the footer button is blocked while the dialog is open');
  shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-consent-dictate-scrolled-viewport.png`});
  // a toast stacks ABOVE the modal (top layer), not under it
  await page.evaluate(() => toast('Probe toast over a dialog'));
  await sleep(200);
  const tt = await page.evaluate(() => { const t = document.querySelector('#toast'), r = t.getBoundingClientRect(); const h = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return {open: t.matches(':popover-open'), top: !!h && (h === t || t.contains(h)), w: r.width}; });
  check(tt.open && tt.w > 40, 'a toast shown while a dialog is open is in the top layer, shown after the modal (hit tests skip it: the page behind is inert; the screenshot shows it above)');
  shot++; await page.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-toast-over-dialog.png`});
  await page.evaluate(() => window.scrollTo(0, 0));
  await words('the consent card'); await noScroll('the consent card'); await snap('consent-dictate');
  await ticks.nth(1).check();
  await closeDlg();                                                      // "Not now" = the first button
  t = await cardText('dictate-stub');
  check(/Not asked yet/.test(t) && await card('dictate-stub').getAttribute('data-state') === 'stopped', '"Not now" changed nothing: still not asked, still stopped');
  await card('dictate-stub').locator('[data-act=consent]').click();
  await page.waitForSelector('#sd[open] .consent', {timeout: 5000});
  await page.locator('#sd .pri').click();
  await waitToast(/is on/, 'the plugin turned on');
  check(!!await until(async () => (await card('dictate-stub').getAttribute('data-state')) === 'running', 'the plugin runs', 25000), 'Allow and turn on: the plugin runs');
  t = await cardText('dictate-stub');
  check(/Allowed/.test(t) && /3 of 3 allowed/.test(t) && /Running/.test(t), 'the card says Allowed, 3 of 3, Running: ' + t.slice(0, 200));
  await snap('dictate-running');

  // ---- 6. the consent card (free plugin with the sensitive permissions)
  console.log(`${tag} consent card, speech + network`);
  await card('voice-demo').locator('[data-act=consent]').click();
  await page.waitForSelector('#sd[open] .consent', {timeout: 5000});
  const vrows = await page.locator('#sd .consent li').evaluateAll(es => es.map(e => ({t: e.innerText.replace(/\s+/g, ' ').trim(), care: e.classList.contains('care'), box: !!e.querySelector('input')})));
  const has = re => vrows.find(r => re.test(r.t));
  check(has(/Turn speech into text on this computer \(nothing leaves it\)/) && has(/Turn text into speech on this computer/), 'speech.listen and speech.speak have their own words');
  check(has(/Accept connections from your home network/) && has(/Hear what this computer plays \(a call, a video\)/), 'net.lan and device.audio-capture have their own words');
  check(has(/Reach api\.example\.com on the internet \(nothing else\)/), 'a scoped host is named, with "nothing else"');
  check(has(/Accept wyoming connections from your home network on port 15310/) && has(/follows from what you allow above/), 'the port it listens on is a line of its own, not a tick box');
  check(has(/read-only overlay page/), 'its overlay is a line of its own');
  check(vrows.filter(r => r.box).length === 6 && vrows.length === 8, '6 tick boxes + 2 follow-on lines: ' + vrows.length + ' rows');
  check(vrows.filter(r => r.care).length === 4 && !has(/Turn text into speech/).care && !has(/Turn speech into text/).care, 'only what senses or reaches out is marked (audio capture, home network, internet host, the lan port); local speech is not');
  check(/stays on this computer/.test(has(/Turn speech into text/).t) && !/reaches beyond PXA/.test(has(/Turn speech into text/).t) && !/reaches beyond PXA/.test(has(/Turn text into speech/).t), 'speech.listen and speech.speak are tagged "stays on this computer", never "reaches beyond PXA"');
  check(/uses a sensor/.test(has(/Hear what this computer plays/).t) && /reaches beyond PXA/.test(has(/Accept connections from your home network/).t) && /reaches beyond PXA/.test(has(/Reach api\.example\.com/).t), 'audio capture is "uses a sensor"; the home network and the internet host are "reaches beyond PXA"');
  check(/reaches beyond PXA/.test(await dlgText()) && /uses a sensor/.test(await dlgText()), 'a marked row says so in words, not colour alone');
  await words('the voice consent card'); await noScroll('the voice consent card'); await snap('consent-voice');
  await closeDlg();
  check(/Not asked yet/.test(await cardText('voice-demo')), 'Not now on the voice plugin changed nothing');
  await card('voice-demo').locator('[data-act=consent]').click();
  await page.waitForSelector('#sd[open] .consent', {timeout: 5000});
  await page.locator('#sd .pri').click();
  check(!!await until(async () => (await card('voice-demo').getAttribute('data-state')) === 'running', 'the voice plugin runs', 25000), 'the free plugin runs after Allow');

  // ---- 7. the plugin's page in a sealed frame
  console.log(`${tag} the page, the bridge`);
  await card('voice-demo').locator('[data-act=page]').click();
  await page.waitForSelector('#ph-page iframe.phframe', {timeout: 10000});
  const fe = page.locator('#ph-page iframe.phframe');
  check(await fe.getAttribute('sandbox') === 'allow-scripts', 'the frame is sandbox="allow-scripts" (no allow-same-origin)');
  const fr = await until(() => page.frames().find(f => /\/api\/plugins\/host\/frame/.test(f.url())), 'the frame loaded', 10000);
  if (fr) {
    await fr.waitForSelector('#hi', {timeout: 10000});
    check(!!await until(async () => /bridge ready, theme (dark|light)/.test(await fr.locator('#hi').innerText()), 'the bridge said hello', 10000), 'the bridge is ready: ' + await fr.locator('#hi').innerText());
    check((await fr.locator('#hi').innerText()).endsWith(THEME), 'the page got the Control theme (' + THEME + ')');
    const origin = await fr.evaluate(() => window.origin);
    check(origin === 'null', 'the frame runs in an opaque origin (window.origin is "null"): ' + origin);
    const bg = await fr.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const want = await page.evaluate(() => { const p = document.createElement('i'); p.style.background = 'var(--surface)'; document.body.append(p); const c = getComputedStyle(p).backgroundColor; p.remove(); return c; });
    check(bg === want, 'the page wears Control colours (surface ' + bg + ')');
    await fr.locator('#b-echo').click();
    check(!!await until(async () => /echo ok/.test(await fr.locator('#out').innerText()), 'echo answered'), 'the bridge reaches its own route: ' + norm(await fr.locator('#out').innerText()).slice(0, 140));
    const out1 = await fr.locator('#out').innerText();
    check(/"method":\s*"POST"/.test(out1) && /hello/.test(out1) && /x=1/.test(out1), 'method, query and body crossed intact');
    await fr.locator('#b-other').click();
    check(!!await until(async () => /refused/.test(await fr.locator('#out').innerText()), 'other plugin answered'), 'another plugin is refused: ' + norm(await fr.locator('#out').innerText()));
    await fr.locator('#b-ctl').click();
    check(!!await until(async () => /refused/.test(await fr.locator('#out').innerText()), 'control answered'), 'Control\'s own routes are refused: ' + norm(await fr.locator('#out').innerText()));
    await fr.locator('#b-same').click();
    check(!!await until(async () => /direct fetch/.test(await fr.locator('#out').innerText()), 'same-origin probe answered'), 'probe finished');
    const probe = await fr.locator('#out').innerText();
    check(/cookie blocked/.test(probe) && /storage blocked/.test(probe) && /parent blocked/.test(probe) && /direct fetch blocked/.test(probe), 'no cookie, no storage, no parent, no direct fetch: ' + norm(probe));
    await fr.locator('#b-bin').click();
    check(!!await until(async () => /^bin 200 audio\/wav b64=UklGRgABAgM=/.test(await fr.locator('#out').innerText()), 'binary answered'), 'a binary answer crosses as base64: ' + norm(await fr.locator('#out').innerText()));
  }
  await words('with the plugin page open'); await noScroll('with the plugin page open'); await snap('plugin-page');
  shot++; await fe.screenshot({path: `${OUT}/${THEME}-${WIDTH}-${String(shot).padStart(2, '0')}-plugin-frame-only.png`});
  const rowsAudit = (await hook('/audit', {})).rows;
  check(rowsAudit.some(r => r.plugin === 'voice-demo' && r.event === 'consent') && rowsAudit.some(r => r.plugin === 'voice-demo' && r.event === 'start'), 'the audit log has the consent and the start');
  await page.locator('#ph-page .phbar button').click();
  check(await page.locator('#ph-page iframe').count() === 0, 'Close removes the frame');

  // ---- 8. turn off, withdraw
  console.log(`${tag} turn off, withdraw`);
  await card('voice-demo').locator('[data-act=stop]').click();
  check(!!await until(async () => (await card('voice-demo').getAttribute('data-state')) === 'stopped', 'stopped'), 'Turn off: the plugin stops');
  await card('voice-demo').locator('[data-act=withdraw]').click();
  check(!!await until(async () => (await card('voice-demo').getAttribute('data-consent')) === 'revoked', 'withdrawn'), 'Withdraw consent: the card says Withdrawn, and offers to ask again');
  check(await card('voice-demo').locator('[data-act=consent]').count() === 1, 'the Allow button is back');

  // ---- 9. REVOKED
  console.log(`${tag} revoked`);
  check(await card('dictate-stub').getAttribute('data-state') === 'running', 'the supporter plugin is still running before the owner revokes');
  const rv = await hook('/revoke', {wallet: wid, item: 'dictate'});
  check(rv.ok, 'the owner revoked the grant on the licd side: ' + rv.out);
  const rf = await api('/api/store/refresh', {});
  console.log('  (store refresh answered ' + rf.status + ')');
  check(!!await until(async () => (await card('dictate-stub').getAttribute('data-gate')) === 'revoked', 'the card is revoked at the gate', 30000), 'one sync later the gate says revoked');
  check(!!await until(async () => (await card('dictate-stub').getAttribute('data-state')) !== 'running', 'the plugin stopped', 30000), 'and the plugin was stopped: ' + await card('dictate-stub').getAttribute('data-state'));
  t = await cardText('dictate-stub');
  check(/No longer unlocked/.test(t) && /no longer unlocked on this computer/i.test(t), 'the card says "No longer unlocked": ' + t.slice(0, 200));
  check(await card('dictate-stub').locator('[data-act=sync]').count() === 1 && await card('dictate-stub').locator('[data-act=start], [data-act=consent], [data-act=unlock]').count() === 0, 'only "Check with the Store" is offered, never Turn on');
  await words('revoked'); await noScroll('revoked'); await snap('revoked');
  await card('dictate-stub').locator('[data-act=sync]').click();
  await waitToast(/Checked with the Store|revoked|not linked|unlocked/i, 'sync answered');
  await snap('revoked-after-check');

  // ---- 10. a quiet plugin and the Store tab still work
  console.log(`${tag} quiet plugin, the Store`);
  await card('quiet-demo').locator('[data-act=consent]').click();
  await page.waitForSelector('#sd[open] .consent', {timeout: 5000});
  check(/does not ask for any permission/.test(await dlgText()) && !(await page.locator('#sd .pri').isDisabled()), 'a plugin with no permissions says so and can still be allowed');
  await snap('consent-quiet');
  await closeDlg();
  await page.goto(BASE + '/#store');
  await page.waitForSelector('#wl-root .wl', {timeout: 30000});
  await noScroll('the Store tab');

  check(blocked.length >= 1, 'the browser itself refused the frame\'s direct fetch of Control (CSP connect-src none): ' + blocked.length + ' message(s)');
  check(errors.length === 0, 'no page error or console error' + (errors.length ? ': ' + errors.join(' || ') : ''));
  await b.close();
  console.log(failures.length ? `${tag} ${failures.length} FAILED:\n  - ` + failures.join('\n  - ') : `${tag} all checks passed`);
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('CRASH ' + (e.stack || e)); process.exit(2); });
