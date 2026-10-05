// Headless-browser click-through of PXA Control's Encode tab against the fake encoder, fake Hugging Face, fake licence server and
// a fake engine (tests/encode_env.py). Run it with tests/encode-gui-check.py, which starts the environment and the browser:
//     node encode-gui-check.js BASE_URL VIEWPORT_WIDTH OUT_DIR CTL_PORT [MODE]
// Exit code 0 only when every check passed and the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, WIDTH, OUT, CTLPORT, MODE] = process.argv.slice(2);
const CTL = 'http://127.0.0.1:' + CTLPORT;
const GOOD_KEY = 'pxk1.K-TEST0001.' + 'A1b2C3d4'.repeat(4);
const KEY_SECRET = 'A1b2C3d4A1b2C3d4';
const failures = [], errors = [];
let notices = 0;
let page, shot = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function ctl(p) { const r = await fetch(CTL + '/' + p); return r.json(); }
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
const T = 60000;
async function waitText(sel, text, timeout) {
  await page.waitForFunction(([s, t]) => { const e = document.querySelector(s); return !!e && e.innerText.includes(t); }, [sel, text], {timeout: timeout || T});
}
async function text(sel) { return (await page.locator(sel).first().innerText()).replace(/\s+/g, ' ').trim(); }
async function count(sel) { return page.locator(sel).count(); }
async function snap(name) { shot++; await page.screenshot({path: `${OUT}/${WIDTH}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: true}); }
async function api(path, body) {
  return page.evaluate(async ([p, b]) => { const r = await fetch(p, b ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)} : {}); return {status: r.status, body: await r.text()}; }, [path, body]);
}
async function noHorizontalScroll(label) {
  const w = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
  check(w.sw <= w.cw + 1, `${label}: no horizontal page scroll (${w.sw} <= ${w.cw})`);
}
async function installedCli(edition) {
  const r = await ctl('installed');
  const hit = r.installed.find(p => p.startsWith(edition + '/'));
  return hit ? r.root + '/' + hit.replace(/\/pxqe$/, '') : null;
}
async function setPace(delay) {      // slow the fake tools so a job can be paused, cancelled, crashed
  await ctl('engine_delay?s=' + delay);
  for (const ed of ['pro', 'free']) { const d = await installedCli(ed); if (d) await ctl('set_encoder?dir=' + encodeURIComponent(d) + '&delay=' + delay); }
}
async function openTab() {
  await page.goto(BASE + '/#encode');
  await page.waitForSelector('#en-edition', {timeout: T});
}
async function pickSource(repo) {
  await page.fill('#en-src', repo);
  await page.click('#en-inspect');
  await page.waitForSelector('#en-srcinfo, #en-srcerr', {timeout: T});
}
async function toTargetStep() {
  await page.click('#en-next1');
  await page.waitForSelector('#en-cards .tile', {timeout: T});
  await page.waitForSelector('#en-recs .en-opt, #en-nofit, #en-planerr', {timeout: T});
}
async function chooseTier(key) {
  await page.evaluate(() => { const d = document.querySelector('#en-adv'); if (d) d.open = true; });
  await page.selectOption('#en-advtier', key);
  await page.waitForFunction(k => { const r = document.querySelector('input[name=en-tier]:checked'); return !r || r.value === k || true; }, key);
}
async function runChecks() {
  await page.click('#en-next2');
  await page.waitForSelector('#en-checks, #en-checkserr', {timeout: 150000});
  await page.waitForFunction(() => !document.querySelector('#p-encode .spin'), {timeout: 150000});
}

(async () => {
  const browser = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const ctx = await browser.newContext({viewport: {width: +WIDTH, height: +WIDTH < 600 ? 844 : 900}});
  page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => {
    if (m.type() !== 'error') return;
    // the browser's own notice for an API call that answered 4xx (a refusal the page shows as a sentence) or for the planned Control restart
    if (/^Failed to load resource: (the server responded with a status of 4\d\d|net::ERR_CONNECTION_(REFUSED|RESET)|net::ERR_EMPTY_RESPONSE)/.test(m.text())) { notices++; return; }
    errors.push('console: ' + m.text());
  });
  page.on('dialog', d => d.accept());

  console.log('STEP 1: empty start (no encoder anywhere)');
  await openTab();
  check((await text('#en-edition')).includes('none installed'), 'edition badge says none installed');
  check(await count('#en-none') === 1, 'plain "no encoder found" notice');
  check(await page.locator('#en-get').isVisible(), 'Get the encoder panel is open when nothing is installed');
  await snap('empty');
  await noHorizontalScroll('empty');

  console.log('STEP 2: Download Free (sha256 + signature verified, installed, found with no restart)');
  await page.click('#en-getfree');
  await waitText('#en-pkgmsg', 'installed');
  await waitText('#en-edition', 'Free 1.0.0');
  check(true, 'Free package downloaded and installed; badge shows Free 1.0.0');
  check((await ctl('installed')).installed.some(p => p.startsWith('free/')), 'installed under <root>/free/<build_id>/');
  await page.click('#en-pkgclose');

  console.log('STEP 3: source: a Hugging Face repo id');
  await pickSource('tiny/Tiny-Qwen3');
  const info = await text('#en-srcinfo');
  check(info.includes('Qwen3ForCausalLM') && /supported/i.test(info), 'architecture shown and supported');
  check(info.includes('apache-2.0'), 'licence shown');
  check(/KiB|MiB/.test(info), 'size shown');
  await snap('source');
  await pickSource('nope/Missing');
  check((await text('#en-srcerr')).includes('no model called'), 'missing repo: one plain sentence');
  await pickSource('tiny/Gated');
  check((await text('#en-srcerr')).includes('gated'), 'gated repo: plain sentence about the licence and HF_TOKEN');
  await pickSource('tiny/NoDerivs');
  check((await count('#en-licwarn')) === 1, 'no-derivatives licence: warning about not sharing the result');
  await pickSource('tiny/Tiny-Qwen3');

  console.log('STEP 4: target with Free installed (classic only, PXQN locked)');
  await toTargetStep();
  check(await count('#en-cards .tile') >= 3, 'cards listed (2 detected + another machine)');
  const recs = await page.$$eval('#en-recs .en-opt', els => els.map(e => e.dataset.tier));
  check(recs.length >= 1 && recs.every(k => k.startsWith('pxq') && !k.startsWith('pxqn')), 'Free recommends classic PXQ tiers only: ' + recs.join(','));
  const locked = await page.$$eval('#en-locked .en-lock', els => els.map(e => e.dataset.tier + ':' + e.innerText.includes('Supporter feature')));
  check(locked.length === 7 && locked.every(x => x.endsWith(':true')), 'all 7 PXQN tiers shown locked as Supporter feature: ' + locked.length);
  const lockTxt = await text('#en-locked');
  check(/6-bit class/.test(lockTxt) && /KiB|MiB/.test(lockTxt), 'locked tiers show their quality-class label and size');
  check(await count('#en-howpro2') === 1 && await count('#en-havekey') === 1, 'How to get Pro and I have a key sit next to the locked tiers');
  check((await text('#en-recs')).includes('no measured speed'), 'no invented speed: says so when there is no measured cell');
  await snap('target-free');
  await noHorizontalScroll('target-free');

  console.log('STEP 5: I have a key -> Pro install (bad key refused plainly, good key installs, tiers unlock)');
  await page.click('#en-havekey');
  await page.waitForSelector('#en-key', {timeout: T});
  await page.fill('#en-key', 'not-a-key');
  await page.click('#en-getpro');
  await waitText('#toast', 'does not look like a PXA Quantizer key');
  check(true, 'a malformed key is refused with one plain sentence');
  await page.fill('#en-key', GOOD_KEY);
  await page.click('#en-getpro');
  await waitText('#en-pkgmsg', 'Pro encoder');
  await waitText('#en-edition', 'Pro 1.0.0');
  check(true, 'Pro package installed and selected; badge shows Pro');
  check((await ctl('installed')).installed.some(p => p.startsWith('pro/')), 'installed under <root>/pro/<build_id>/');
  await page.click('#en-pkgclose');
  await waitText('#en-lic', 'Licence OK');
  check((await text('#en-lic')).includes('4 encodes left'), 'licence state and encodes left shown for Pro');
  const html = await page.content();
  check(!html.includes(KEY_SECRET), 'the secret half of the key is nowhere in the page');
  check(await count('#en-keymask') === 0 || !(await text('#en-keymask')).includes(KEY_SECRET), 'the key is masked');
  const ls = await page.evaluate(() => JSON.stringify(Object.assign({}, localStorage)));
  check(!ls.includes(KEY_SECRET), 'the key is not in localStorage');
  await page.waitForFunction(() => document.querySelectorAll('#en-recs .en-opt[data-tier^=pxqn]').length > 0, {timeout: T});
  const recs2 = await page.$$eval('#en-recs .en-opt', els => els.map(e => e.dataset.tier));
  check(recs2.some(k => k.startsWith('pxqn')), 'Pro recommends PXQN tiers first: ' + recs2.join(','));
  const lockedPro = await page.$$eval('#en-locked .en-lock', els => els.map(e => e.dataset.tier));
  check(lockedPro.length > 0 && lockedPro.length < 7, 'tiers outside the key\'s plan stay locked with their reason: ' + lockedPro.join(','));
  check((await text('#en-plannote')).includes('PXQN2'), 'the plan note says what the key includes');
  check((await text('#en-locked')).includes('Not included in your plan'), 'the reason reads "Not included in your plan"');
  await snap('target-pro');

  console.log('STEP 5a: Who can load this file: not for a classic tier; for a Pro PXQN tier shown disabled with one line (this encoder cannot lock yet)');
  const tierNow = await page.evaluate(() => EN.tier);
  if (!tierNow.startsWith('pxqn')) check(!(await page.locator('#en-lock').isVisible()), 'a classic tier (' + tierNow + '): no lock choice on the page');
  await chooseTier('pxqn4');
  await page.waitForSelector('#en-lockset', {timeout: T});
  check(await page.locator('#en-lock').isVisible(), 'a Pro PXQN tier: the choice is on the page');
  check((await text('#en-lock legend')) === 'Who can load this file', 'it is titled "Who can load this file"');
  check(await page.$eval('#en-lockset', e => e.disabled), 'locking is off for this encoder: the choice is disabled');
  check(await count('#en-lockset input[type=radio]') === 1 && await page.locator('#en-lockset input[type=radio]').isDisabled(), 'one greyed-out radio: Only me (recommended)');
  check((await text('#en-lockoff')) === 'File locking turns on with the next encoder update', 'one line: File locking turns on with the next encoder update');
  await snap('lock-off');
  await noHorizontalScroll('lock-off');

  console.log('STEP 5b: the Pro encoder cannot find the NVIDIA CUDA libraries -> one button downloads the GPU runtime');
  const rtProDir = await installedCli('pro');
  await ctl('set_encoder?dir=' + encodeURIComponent(rtProDir) + '&needs_runtime=true');
  await ctl('licence_runtime?m=ok&bulk=2500000&delay=0.01');
  await page.click('#en-rescan');
  await page.waitForSelector('#en-rtneed', {timeout: T});
  check((await text('#en-rtneed')).includes('libcublas.so.12, libcusolver.so.11'), 'says exactly which libraries are missing');
  check((await text('#en-lic')).includes('Cannot load here'), 'the licence line says it cannot load here');
  await page.waitForFunction(() => /one time, [0-9.]+ (KiB|MiB|GiB)\)/.test((document.querySelector('#en-getrt') || {}).innerText || ''), {timeout: T});
  check(true, 'the button names the size: ' + (await text('#en-getrt')));
  check(await count('#en-rtsrc') === 0, 'no "GPU libraries from ..." line while they are missing');
  await snap('runtime-need');
  await noHorizontalScroll('runtime-need');
  await chooseTier('pxqn4');
  await runChecks();
  const rtck = await page.$$eval('#en-checks .en-ck', els => els.map(e => e.dataset.check + ':' + (e.innerText.includes('Stop') ? 'stop' : 'x')));
  check(rtck.includes('runtime:stop'), 'the checks stop at the GPU runtime: ' + rtck.join(' '));
  check(await count('#en-ck-getrt') === 1, 'the same button sits in the checks row');
  check(await page.locator('#en-start').isDisabled(), 'Start encoding is blocked until it is installed');
  check(!(await text('#en-checks')).includes('cannot start on this computer. The Pro encoder needs'), 'the problem is said once, not repeated under Tools');
  await snap('runtime-checks');
  await ctl('licence_runtime?m=tamper&bulk=2500000');
  await page.click('#en-ck-getrt');
  await waitText('#en-rtmsg', 'does not match its checksum');
  check(true, 'a tampered runtime is refused in plain words and nothing is installed');
  check((await ctl('runtime_installed')).installed.length === 0, 'no runtime folder, not even a half-installed one');
  check(await count('#en-rtneed') === 1, 'the need (and the button) stays');
  await ctl('licence_runtime?m=ok&bulk=2500000&delay=0.01');
  await page.click('#en-rtretry');
  await page.waitForSelector('#en-rtcancel', {timeout: T});
  check(true, 'the download shows progress and a Cancel button');
  await snap('runtime-downloading');
  await noHorizontalScroll('runtime-downloading');
  await waitText('#en-rtmsg', 'The Pro encoder can start now');
  await page.waitForFunction(() => !document.querySelector('#en-rtneed'), {timeout: T});
  check((await ctl('runtime_installed')).installed.length === 1, 'the runtime is installed once, next to the encoders');
  await page.waitForSelector('#en-rtsrc', {timeout: T});
  check((await text('#en-rtsrc')).includes('PXA GPU runtime'), 'the Encode tab says where the GPU libraries come from: ' + (await text('#en-rtsrc')));
  await page.waitForFunction(() => { const b = document.querySelector('#en-start'); return b && !b.disabled; }, {timeout: 150000});
  check(true, 'the checks ran again by themselves and Start encoding is enabled');
  await snap('runtime-done');
  await noHorizontalScroll('runtime-done');
  await page.click('#en-back3');
  await page.waitForSelector('#en-recs .en-opt, #en-nofit, #en-planerr', {timeout: T});

  console.log('STEP 6: a whole encode, start to finish');
  await setPace(0.12);
  await chooseTier('pxqn4');
  await runChecks();
  const cks = await page.$$eval('#en-checks .en-ck', els => els.map(e => e.dataset.check + ':' + (e.innerText.includes('OK') ? 'ok' : 'x')));
  check(cks.length >= 7 && cks.every(x => x.endsWith(':ok')), 'every check passes: ' + cks.join(' '));
  check(/about how long/i.test(await text('#en-est')), 'time estimate and disk figures shown');
  await snap('checks');
  await noHorizontalScroll('checks');
  await page.click('#en-start');
  await page.waitForSelector('#en-stages [data-stage=download]', {timeout: T});
  check((await count('#en-stages .en-stage')) === 8, 'a progress row per stage (8)');
  await page.waitForFunction(() => { const e = document.querySelector('[data-stage=encode]'); return e && e.classList.contains('running'); }, {timeout: T});
  await snap('running');
  await noHorizontalScroll('running');
  check((await text('#en-jtext')).includes('%'), 'overall progress and ETA text');
  await page.click('#en-logd summary');
  check((await page.locator('#en-log').innerText()).length > 20, 'log drawer shows output');
  await page.waitForSelector('#en-doneok', {timeout: 120000});
  const outp = await text('#en-outpath'), sha = await text('#en-outsha');
  check(outp.endsWith('Tiny-Qwen3-PXQN4.gguf'), 'output path shown: ' + outp.split('/').pop());
  check(/^[0-9a-f]{64}$/.test(sha), 'sha256 shown');
  check((await text('#en-doneok')).includes('6-bit class'), 'quality-class label shown');
  check((await text('#en-lockdone')).includes('Not locked') && (await text('#en-lockdone')).includes('Anyone with a PXA engine can load this file'), 'done screen: this encoder cannot lock yet, so it says the file is not locked and anyone can load it');
  check((await ctl('models')).files.includes('Tiny-Qwen3-PXQN4.gguf'), 'the file is on disk');
  await snap('done');
  await noHorizontalScroll('done');

  console.log('STEP 7: Test it -> the existing server + bench flow, then Share your score');
  await page.click('#en-testit');
  await page.waitForFunction(() => { const e = document.querySelector('#en-testres, #en-testerr'); return !!e; }, {timeout: 240000});
  const tr = await text('#en-test');
  check(tr.includes('It runs') && tr.includes('42.5'), 'Test it started the server, ran the bench: ' + tr.slice(0, 90));
  await page.waitForSelector('#en-share:not([hidden])', {timeout: 5000});
  await page.click('#en-share');
  await page.waitForSelector('#hs:not([hidden])', {timeout: 30000});
  check(true, 'Share your score opens the existing high-score panel');
  const hsBody = await page.inputValue('#hs-body');
  check(hsBody.includes('Tiny-Qwen3-PXQN4.gguf') && !hsBody.includes(KEY_SECRET), 'score payload names the new file and carries no key');
  await snap('share');
  await page.click('#hs-skip');
  await page.evaluate(() => showTab('encode'));
  await page.waitForSelector('#en-doneok', {timeout: T});

  console.log('STEP 7b: a lock-capable encoder and a licence server with locking on: Who can load this file');
  const enc = x => encodeURIComponent(x);
  const lockDir = await installedCli('pro');
  const LOCK_SUP = JSON.stringify({enabled: true, allowed_modes: ['personal', 'supporters'], default_mode: 'personal', epoch: '2026-10'});
  const LOCK_ME = JSON.stringify({enabled: true, allowed_modes: ['personal'], default_mode: 'personal', epoch: '2026-10'});
  const LOCK_OFF = JSON.stringify({enabled: false, allowed_modes: ['open'], default_mode: 'open', epoch: '2026-10'});
  const lockMakes = async () => Object.values(await ctl('state')).flatMap(x => x.makes || []).map(m => m.argv).filter(a => a.includes('--lock')).map(a => a[a.indexOf('--lock') + 1]);
  const rescanPlan = async () => { await page.click('#en-rescan'); await page.waitForFunction(() => !document.querySelector('#en-tiers .spin') && document.querySelector('#en-recs .en-opt, #en-nofit, #en-planerr'), {timeout: T}); };
  // the Rescan re-plans in the background: wait for the lock box to show what the new answer says (the old box has the same selectors)
  const waitLockOn = n => page.waitForFunction(k => { const f = document.querySelector('#en-lockset'); return !!f && f.dataset.state === 'on' && document.querySelectorAll('#en-lockopts .en-opt').length === k; }, n, {timeout: T});
  const waitLockOff = () => page.waitForSelector('#en-lockset[data-state=off]', {timeout: T});
  await setPace(0.02);
  await ctl('set_encoder?dir=' + enc(lockDir) + '&lock_support=true&lock_server=' + enc(LOCK_SUP));
  await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await rescanPlan();                                                  // the encoder changed on disk: Rescan reads it again, and the plan with it
  await chooseTier('pxqn4');
  await waitLockOn(2);
  const labels = await page.$$eval('#en-lockopts .en-opt', els => els.map(e => e.querySelector('.t1').innerText.trim()));
  check(labels.join(' | ') === 'Only me (recommended) | Any PXA supporter', 'the plan allows two modes: ' + labels.join(' | ') + ' (no "Anyone": the tier does not allow it)');
  check(!(await page.$eval('#en-lockset', e => e.disabled)) && await page.locator('#en-lockopts input[value=personal]').isChecked(), 'enabled, and "Only me (recommended)" is the default');
  check((await text('#en-lockneeds')).includes('v3 or newer'), 'says a locked file loads in PXA v3 or newer');
  await snap('lock-on');
  await noHorizontalScroll('lock-on');
  await page.check('#en-lockopts input[value=supporters]');
  await runChecks();
  const lockRow = await text('#en-checks [data-check=lock]');
  check(lockRow.includes('OK') && lockRow.includes('Any active PXA supporter will be able to load this file'), 'the checks say who will be able to load it: ' + lockRow.slice(0, 90));
  await snap('lock-checks');
  await noHorizontalScroll('lock-checks');
  await page.click('#en-start');
  await page.waitForSelector('#en-doneok', {timeout: 120000});
  const dsup = await text('#en-lockdone');
  check(dsup.includes('Locked to PXA supporters') && dsup.includes('Any active PXA supporter can load this file'), 'done screen: locked, and who can load it: ' + dsup.slice(0, 80));
  check(dsup.includes('loads in PXA v3 or newer'), 'done screen: loads in PXA v3 or newer');
  check((await lockMakes()).includes('supporters'), 'the encoder was run with --lock supporters (and nothing else carries the choice)');
  await snap('done-locked-supporters');
  await noHorizontalScroll('done-locked-supporters');
  // the default (nothing touched) is personal
  await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await chooseTier('pxqn3');
  await waitLockOn(2);
  check(await page.locator('#en-lockopts input[value=supporters]').isChecked(), 'the last choice is remembered');
  await page.check('#en-lockopts input[value=personal]');
  await runChecks();
  await page.click('#en-start');
  await page.waitForSelector('#en-doneok', {timeout: 120000});
  const dme = await text('#en-lockdone');
  check(dme.includes('Locked to you') && dme.includes('Only you can load this file') && dme.includes('loads in PXA v3 or newer'), 'done screen: locked to you: ' + dme.slice(0, 80));
  check((await lockMakes()).slice(-1)[0] === 'personal', 'the encoder was run with --lock personal');
  await snap('done-locked-personal');
  // a classic tier has no lock choice
  await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await chooseTier('pxq3');
  check(!(await page.locator('#en-lock').isVisible()), 'a classic tier: no "Who can load this file"');
  // a plan that allows one mode: one radio; and the server's switch off again: disabled, one line, and the file is written open
  await ctl('set_encoder?dir=' + enc(lockDir) + '&lock_server=' + enc(LOCK_ME));
  await rescanPlan();
  await chooseTier('pxqn4');
  await waitLockOn(1);
  check((await text('#en-lockopts')).startsWith('Only me (recommended)') && !(await page.$eval('#en-lockset', e => e.disabled)), 'a tier that allows only a personal lock shows one choice, enabled');
  await ctl('set_encoder?dir=' + enc(lockDir) + '&lock_server=' + enc(LOCK_OFF));
  await rescanPlan();
  await chooseTier('pxqn4');
  await waitLockOff();
  check(await page.$eval('#en-lockset', e => e.disabled) && (await text('#en-lockoff')) === 'File locking turns on with the next encoder update', 'the switch is off: disabled, with the one line');
  await snap('lock-off-server');
  await noHorizontalScroll('lock-off-server');
  await runChecks();
  check((await text('#en-checks [data-check=lock]')).includes('without a lock'), 'the checks say the file will be written without a lock');
  await page.click('#en-start');
  await page.waitForSelector('#en-doneok', {timeout: 120000});
  check((await text('#en-lockdone')).includes('Not locked') && (await text('#en-lockdone')).includes('Anyone with a PXA engine can load this file'), 'done screen: not locked, anyone can load it');
  check((await lockMakes()).slice(-1)[0] === 'open', 'the encoder was run with --lock open');
  await snap('done-open');
  await noHorizontalScroll('done-open');
  // a refusal from the licence server about the lock, in plain words (never a flag, an HTTP code or a revoked key)
  await ctl('set_encoder?dir=' + enc(lockDir) + '&lock_server=' + enc(LOCK_SUP) + '&lock_refuse=' + enc('"your tier may not write a supporters lock; the modes it may use: personal"'));
  await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await rescanPlan();
  await chooseTier('pxqn4');
  await waitLockOn(2);
  await runChecks();
  await page.click('#en-start');
  await waitText('#en-jstatus', 'stopped with an error', 60000);
  const lerr = await text('#en-jerr');
  check(lerr.includes('Your plan does not allow a file that any PXA supporter can load') && lerr.includes('It allows: only me.'), 'a lock refusal is one plain sentence: ' + lerr.slice(0, 110));
  check(!/--lock|HTTP|revoked|Traceback/.test(lerr), 'no flag, HTTP code, revoked key or trace in it');
  await snap('lock-refused');
  await noHorizontalScroll('lock-refused');
  await ctl('set_encoder?dir=' + enc(lockDir) + '&lock_support=false&lock_server=null&lock_refuse=null');       // back to the encoder the later steps expect
  await page.click('#en-rescan');
  await page.waitForFunction(() => !document.querySelector('#p-encode .spin'), {timeout: T});
  await page.click('#en-discard');

  console.log('STEP 8: pause, resume, cancel, resume');
  if (await count('#en-another')) await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await setPace(0.25);
  await chooseTier('pxqn3');
  await runChecks();
  await page.click('#en-start');
  await page.waitForFunction(() => { const e = document.querySelector('[data-stage=reference]'); return e && (e.classList.contains('running') || e.classList.contains('done')); }, {timeout: T});
  await page.click('#en-pause');
  await waitText('#en-jstatus', 'paused');
  const p1 = await text('#en-jtext'); await sleep(2500); const p2 = await text('#en-jtext');
  check(p1 === p2, 'paused: progress does not move (' + p1 + ')');
  await snap('paused');
  await page.click('#en-resume');
  await waitText('#en-jstatus', 'running');
  await sleep(800);
  await page.click('#en-cancel');
  await waitText('#en-jstatus', 'cancelled');
  check((await text('#en-jerr')).includes('Cancelled'), 'cancel: plain sentence, finished steps kept');
  await snap('cancelled');
  await setPace(0.02);
  await page.click('#en-resume');
  await waitText('#en-jstatus', 'finished', 120000);
  check((await ctl('models')).files.includes('Tiny-Qwen3-PXQN3.gguf'), 'resumed after cancel and finished');

  console.log('STEP 9: resume after a reboot (Control killed mid-encode)');
  await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await setPace(0.3);
  await chooseTier('pxqn5');
  await runChecks();
  const before = JSON.stringify((await ctl('state')));
  await page.click('#en-start');
  await page.waitForFunction(() => { const e = document.querySelector('[data-stage=hessians]'); return e && e.classList.contains('running'); }, {timeout: T});
  await sleep(1500);
  await ctl('restart?kill=1');
  await page.goto(BASE + '/#encode');
  await page.reload();
  await page.waitForSelector('#en-jstatus', {timeout: T});
  await waitText('#en-jstatus', 'interrupted');
  check((await text('#en-jerr')).includes('closed'), 'after the restart the job is shown as interrupted with a plain sentence');
  await snap('interrupted');
  await setPace(0.01);
  await page.click('#en-resume');
  await waitText('#en-jstatus', 'finished', 120000);
  const after = await ctl('state');
  const st = Object.values(after).find(x => x.jobs !== undefined);
  check(st && Array.isArray(st.resumes) && st.resumes.length >= 1, 'the licence job was RESUMED (no second charge): ' + JSON.stringify(st));
  check((await ctl('models')).files.includes('Tiny-Qwen3-PXQN5.gguf'), 'the resumed job finished and delivered the file');

  console.log('STEP 10: refusals in plain words (never a stack trace)');
  const proDir = await installedCli('pro');
  await page.click('#en-another');
  await pickSource('tiny/Tiny-Qwen3');
  await toTargetStep();
  await chooseTier('pxqn4');
  await ctl('set_encoder?dir=' + encodeURIComponent(proDir) + '&licence_check=' + encodeURIComponent(JSON.stringify({state: 'valid', user: 'tester', encodes_left: 0, expires: '2026-12-01'})));
  await runChecks();
  check((await text('#en-refuse')).includes('no encodes left') && await page.locator('#en-start').isDisabled(), 'zero encodes left: refused with a sentence, Start disabled');
  await ctl('set_encoder?dir=' + encodeURIComponent(proDir) + '&licence_check=' + encodeURIComponent(JSON.stringify({state: 'revoked', user: 'tester', encodes_left: 3})));
  await page.click('#en-recheck');
  await page.waitForFunction(() => !document.querySelector('#p-encode .spin'), {timeout: 60000});
  check((await text('#en-checks')).includes('revoked'), 'revoked key: plain sentence about the licence server');
  await snap('refused');
  await ctl('set_encoder?dir=' + encodeURIComponent(proDir) + '&licence_check=' + encodeURIComponent(JSON.stringify({state: 'valid', user: 'tester', encodes_left: 3})) + '&fail=%22offline%22');
  await page.click('#en-recheck');
  await page.waitForFunction(() => !document.querySelector('#p-encode .spin'), {timeout: 60000});
  await page.click('#en-start');
  await waitText('#en-jstatus', 'stopped with an error', 60000);
  check((await text('#en-jerr')).includes('Cannot reach the licence server'), 'offline during the run: plain sentence + hint');
  check(!(await text('#en-jerr')).includes('Traceback'), 'no traceback on the page');
  await ctl('set_encoder?dir=' + encodeURIComponent(proDir) + '&fail=null');
  await page.click('#en-resume');
  await waitText('#en-jstatus', 'finished', 120000);
  check(true, 'after the licence server is back, Resume finishes the job');

  console.log('STEP 11: an update is offered and installs in one click');
  await ctl('licence_new_build?free=b-free-3&pro=b-pro-3');
  await ctl('set_encoder?dir=' + encodeURIComponent(proDir) + '&licence_check=null');
  await api('/api/encode/update', {force: true});
  await page.reload();
  await page.waitForSelector('.en-upd', {timeout: T});
  check((await text('.en-upd')).includes('Update available'), 'Update available is shown');
  await page.click('.en-upd button');
  await waitText('#en-pkgmsg', 'installed');
  check((await ctl('installed')).installed.some(p => p.includes('b-pro-3')), 'the update installed next to the old build');

  console.log('STEP 12: bad downloads are refused');
  for (const [mode, needle] of [['bad_signature', 'signature'], ['bad_checksum', 'damaged'], ['revoked', 'revoked'], ['noquota', 'no encodes left'], ['server_error', 'having trouble'], ['invalid_key', 'does not recognise'], ['wrong_platform', 'does not match'], ['bad_manifest_file', 'does not match its checksum'], ['link_expired', 'expired']]) {
    await ctl('licence_mode?m=' + mode);
    await api('/api/encode/get', {edition: 'pro'});
    await page.waitForFunction(() => { const e = document.querySelector('#en-pkgmsg'); return !!e && /refused|damaged|revoked|no encodes|trouble|signature/.test(e.innerText); }, {timeout: T}).catch(() => {});
    await page.reload();
    await page.waitForSelector('#en-edition', {timeout: T});
    const m = await api('/api/encode/state');
    const j = JSON.parse(m.body);
    check(j.pkg.phase === 'failed' && j.pkg.message.toLowerCase().includes(needle), `${mode}: ${j.pkg.message.slice(0, 80)}`);
    check(!(j.pkg.message.includes(KEY_SECRET)), `${mode}: message carries no key`);
  }
  await ctl('licence_mode?m=ok');

  console.log('STEP 13: nothing sensitive leaked');
  const reqs = await ctl('requests');
  const sensitive = r => JSON.stringify(r).includes(KEY_SECRET);
  check(!reqs.hf.some(sensitive) && !reqs.cdn.some(sensitive) && !reqs.board.some(sensitive), 'the key never reached Hugging Face, the CDN or the score board');
  check(reqs.lic.every(r => !r.path.includes(KEY_SECRET) && !JSON.stringify(r.headers).includes(KEY_SECRET)), 'the key only travels in the body of the licence server POSTs');
  check(reqs.lic.some(r => r.method === 'POST' && r.body.includes(KEY_SECRET)), '(and it did reach the licence server)');
  const rb = await api('/api/report/bundle');
  check(rb.status === 200 && !rb.body.includes(KEY_SECRET), 'Report a problem bundle carries no key');
  const jobs = JSON.parse((await api('/api/encode/state')).body).jobs;
  let logs = '';
  for (const j of jobs) logs += (await api('/api/encode/log?id=' + j.id)).body;
  check(logs.length > 100 && !logs.includes(KEY_SECRET), 'job logs carry no key (the fake encoder printed it once on purpose: ' + (logs.includes('<key>') ? 'scrubbed' : 'not printed') + ')');
  const cfg = await ctl('config');
  check(cfg.encode && cfg.encode.licence_key === GOOD_KEY, 'the key is stored in the Control config (and only there)');

  console.log('PAGE ERRORS:', errors.length ? errors.join(' | ') : 'none', '(browser notices for handled 4xx refusals and the planned restart: ' + notices + ')');
  check(errors.length === 0, 'zero page errors');
  await browser.close();
  console.log(failures.length ? 'RESULT: FAIL (' + failures.length + ')' : 'RESULT: PASS');
  process.exit(failures.length ? 1 : 0);
})().catch(async e => { console.log('SCENARIO CRASHED:', e.message); try { await snap('crash'); } catch (x) { /* */ } console.log('PAGE ERRORS:', errors.join(' | ')); process.exit(2); });
