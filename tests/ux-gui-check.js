// Headless-browser check of the PXA Control UX revamp (2026-10). Run it with tests/ux-gui-check.py, which starts the
// environment for each mode and the browser:   node ux-gui-check.js BASE MODE THEMES WIDTHS OUT_DIR
// MODE first | ready | serving (see ux-gui-check.py). Exit code 0 only when every check passed and no page error was logged.
const { chromium } = require('playwright-core');
const [BASE, MODE, THEMES, WIDTHS, OUT] = process.argv.slice(2);
const failures = [], sleep = ms => new Promise(r => setTimeout(r, ms));
let page, tag = '', shot = 0, shotStart;
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(tag + ' ' + msg); console.log('  FAIL ' + msg); } }
async function snap(name, full) { shot++; await page.screenshot({path: `${OUT}/${MODE}-${tag}-${String(shot).padStart(2, '0')}-${name}.png`, fullPage: full !== false}); }
const count = sel => page.locator(sel).count();
const txt = async sel => (await page.locator(sel).allInnerTexts()).join(' | ').replace(/\s+/g, ' ').trim();
const visible = async sel => (await count(sel)) > 0 && await page.locator(sel).first().isVisible();
async function tab(t) { await page.locator(`nav.tabs button[data-tab=${t}]`).click(); await sleep(900); }

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  for (const THEME of THEMES.split(',')) for (const WIDTH of WIDTHS.split(',')) {
    const W = +WIDTH, phone = W <= 820; tag = `${THEME}-${W}`; shot = 0;
    const errors = [], failed = [];
    const ctx = await b.newContext({viewport: {width: W, height: phone ? 844 : 900}, colorScheme: THEME, isMobile: phone, hasTouch: phone});
    page = await ctx.newPage();
    // serving mode walks the Plugins / Store PREVIEW design (store-mock.json: 8 items, 40 credits, Anibal owned). Since the
    // store backend (the Control hardening pass + the store pass) /api/info always announces the store routes, so the page would read
    // the backend instead; hiding the announcement here keeps the preview path covered. first/ready use the backend.
    if (MODE === 'serving') await page.route('**/api/info', async r => {
      const resp = await r.fetch(); const j = await resp.json(); delete j.store; await r.fulfill({response: resp, json: j});
    });
    page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
    page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push('console: ' + m.text()); });
    page.on('response', r => { if (r.status() >= 400 && r.url().startsWith(BASE)) failed.push(r.status() + ' ' + r.url().slice(BASE.length)); });
    console.log(`[${MODE} ${THEME} ${W}px] Home is the landing page and follows the system theme`);
    await page.goto(BASE + '/');
    await page.waitForSelector('#home-root .hero', {timeout: 45000});
    await sleep(2500);
    check(await count('nav.tabs button[data-tab=home][aria-selected=true]') === 1, 'a fresh visit opens Home');
    check(await page.evaluate(() => document.documentElement.dataset.theme) === THEME, 'no saved choice: the page follows the OS theme (' + THEME + ')');
    const geo = await page.evaluate(() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}));
    check(geo.sw <= geo.cw + 1, `no horizontal page scroll (${geo.sw} <= ${geo.cw})`);
    if (MODE === 'first') {
      check((await txt('.hstep.now h4')).startsWith('Get a model'), 'first run: step 1 "Get a model" is the current step');
      check(await count('.hstep.later') === 2, 'first run: steps 2 and 3 wait their turn');
      check(await visible('#h-folder') && await visible('#h-folder-add'), 'first run: the folder path is offered');
      const dl = await txt('.hopt');
      check(/Download one from PXA/.test(dl) && (/Download/.test(dl) || /Can't reach huggingface\.co|Reading the PXA model list/.test(dl)), 'first run: downloads are offered, or the page says why not: ' + dl.slice(0, 120));
      await snap('home-first');
      await page.locator('#h-folder-add').click(); await sleep(300);
      check(/folder/i.test(await txt('#toast')), 'an empty folder field explains itself instead of doing nothing: ' + (await txt('#toast')).slice(0, 80));
    } else if (MODE === 'ready') {
      check((await txt('.hstep.now h4')).startsWith('Pick cards and start'), 'ready: step 2 is the current step, step 1 is done');
      check(await visible('#h-start') && await page.locator('#h-start').isEnabled(), 'ready: one enabled Start button');
      check(await count('#h-model option') >= 1, 'ready: the model picker lists the library');
      check(await count('.cardsum .chip') >= 1, 'ready: the cards it will start on are named');
      await snap('home-ready');
      if (shotStart === undefined) {                    // once per run: Start goes through Launch's doStart, and its answer lands on Home
        shotStart = true;
        await page.locator('#h-start').click();
        await page.waitForSelector('#home-root .alert.bad, #home-root .srvrow', {timeout: 120000});
        const said = await txt('#home-root .alert.bad, #home-root .srvrow');
        check(said.length > 0, 'Start answers on Home: a running row, or a refusal in words with See why: ' + said.slice(0, 140));
        await snap('home-start-answer');
      }
    } else {
      check(await count('.srvrow') === 2, 'serving: every running server is listed (' + await count('.srvrow') + ')');
      check(await count('.srvrow .b.pri') >= 1 && /Chat/.test(await txt('.srvrow .b.pri')), 'serving: Chat is the main action');
      await snap('home-serving');
    }
    if (phone) {
      const nav = await page.evaluate(() => [...document.querySelectorAll('nav.tabs button[role=tab]')].slice(0, 5).map(b => { const r = b.getBoundingClientRect(); return [b.dataset.tab, r.left >= -1 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1 && r.top > innerHeight / 2]; }));
      check(nav.every(x => x[1]), 'phone: the five main sections sit in the bottom bar, on screen: ' + nav.map(x => x[0]).join(','));
    } else {
      check(await count('nav.tabs .navlbl') >= 4, 'desktop: the sections are grouped (Run / Make and tune / Watch / Extend)');
    }

    console.log(`[${MODE} ${THEME} ${W}px] theme: one click flips it, the Help menu offers System`);
    await page.click('#theme'); await sleep(300);
    const flipped = await page.evaluate(() => document.documentElement.dataset.theme);
    check(flipped !== THEME, 'the sun button flips the theme (' + THEME + ' -> ' + flipped + ')');
    await page.click('#help-btn'); await sleep(300);
    check(await visible('#help-menu') && await visible('#rp-open'), 'the Help menu holds Report a problem, community and theme');
    await snap('help-menu', false);
    await page.click('#theme-pref button[data-v=system]'); await sleep(300);
    check(await page.evaluate(() => document.documentElement.dataset.theme) === THEME, 'System goes back to following the OS');
    await page.keyboard.press('Escape'); await sleep(200);

    if (MODE === 'serving' || (MODE === 'first' && !phone)) {
      console.log(`[${MODE} ${THEME} ${W}px] every section opens`);
      for (const t of ['chat', 'launch', 'servers', 'models', 'encode', 'speed', 'profiles', 'live', 'rig', 'plugins', 'store']) {
        await tab(t);
        check(await count(`nav.tabs button[data-tab=${t}][aria-selected=true]`) === 1 && await visible(`#p-${t}`), t + ' opens');
        if (MODE === 'serving' || ['servers', 'models', 'chat'].includes(t)) await snap(t, t !== 'live' && t !== 'profiles');
      }
    }
    if (MODE === 'serving') {
      console.log(`[${MODE} ${THEME} ${W}px] plugins: Anibal leads, consent before it runs`);
      await tab('plugins');
      await page.waitForSelector('.phero', {timeout: 15000});
      check(/Anibal/.test(await txt('.phero h3')), 'Anibal is the hero card');
      check(/Requested on Discord/.test(await txt('#pl-root')), 'plugins say where the request came from');
      check(await count('#p-plugins a[href="https://discord.pxanetwork.com"]') >= 1, 'a Request a plugin link to the Discord');
      check(/Preview/.test(await txt('#pl-root .alert.info')), 'preview data says it is a preview');
      await page.locator('.phero .sw').click(); await sleep(400);
      check(await page.evaluate(() => document.getElementById('sd').open), 'turning Anibal on asks first (consent dialog)');
      check(/What it can reach/.test(await txt('#sd')) && /Where it can send data/.test(await txt('#sd')), 'the consent lists what it can reach and where data goes');
      await snap('plugin-consent', false);
      await page.locator('#sd-acts .b.pri').click(); await sleep(500);
      check(await page.locator('.phero input[role=switch]').isChecked(), 'approved: Anibal is on');
      await snap('plugins-on');

      console.log(`[${MODE} ${THEME} ${W}px] store: credits, kinds, locked states, activation`);
      await tab('store');
      check((await txt('#st-credits .bal')).startsWith('40'), 'the balance is shown: ' + await txt('#st-credits .bal'));
      check(/donation/i.test(await txt('#st-credits')), 'credits say they come from donations');
      check(await count('#st-root .pcard') === 8, 'the catalogue lists every item (' + await count('#st-root .pcard') + ')');
      const kinds = await page.evaluate(() => [...new Set([...document.querySelectorAll('#st-root .pcard .kind')].map(e => e.textContent))].sort().join(','));
      check(kinds === 'Feature,Plugin,Preset,Quant tier', 'four kinds: ' + kinds);
      check(await count('#st-root .pcard.locked') >= 1 && await count('#st-root .pcard.active') >= 1, 'locked and active states are both drawn');
      await snap('store');
      await page.locator('#st-root .kfilter button', {hasText: 'Quant tiers'}).click(); await sleep(300);
      check(await count('#st-root .pcard') === 1, 'the kind filter narrows the catalogue');
      await page.locator('[data-item="pxqn-tier"] .b.sup').click(); await sleep(300);
      check(await page.evaluate(() => document.getElementById('sd').open) && /30 coins/.test(await txt('#sd')), 'Activate asks first and names the price');
      await snap('store-activate', false);
      await page.locator('#sd-acts .b.pri').click(); await sleep(600);
      check((await txt('#st-credits .bal')).startsWith('10'), 'after activation the balance drops (preview): ' + await txt('#st-credits .bal'));
      check(await count('[data-item="pxqn-tier"].active') === 1, 'the PXQN tier reads active');
      await page.locator('#st-root .kfilter button', {hasText: 'Everything'}).click(); await sleep(200);
      await page.goto(BASE + '/?store=offline#store'); await sleep(2500);
      check(/can't be reached/i.test(await txt('#st-root .alert.warn')), 'offline within the grace period: the page says what still works');
      await snap('store-offline', false);
      await page.goto(BASE + '/?store=expired#store'); await sleep(2500);
      check(/Offline too long/i.test(await txt('#st-root .alert.bad')), 'offline past the grace period: paused, and why');
    }
    const ours = failed.filter(f => /store|home\.js|plugins\.js|\/api\/info/.test(f));
    check(ours.length === 0, 'the new files and the store adapter made no failing request' + (ours.length ? ': ' + ours.join(' | ') : ''));
    if (failed.length) console.log('  note: failing requests of the environment (not counted): ' + [...new Set(failed)].slice(0, 6).join(' | '));
    check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
    await ctx.close();
  }
  await b.close();
  if (failures.length) { console.log('FAILURES:'); failures.forEach(f => console.log(' - ' + f)); process.exit(1); }
  console.log('all checks passed (' + MODE + ')');
})().catch(e => { console.log('FAIL ' + (e.stack || e.message)); process.exit(1); });
