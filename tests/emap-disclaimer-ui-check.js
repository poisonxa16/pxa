// Headless-browser check that PXA Control's Expert map card shows the server's own words about what the map
// is and is not (GET /api/expert-map -> "disclaimer", defined in tools/pxa_expert_map.py). Run it with
// tests/emap-disclaimer-ui-check.py, which starts the hermetic environment of tests/encode_env.py and the
// browser:
//     node emap-disclaimer-ui-check.js BASE_URL OUT_DIR
// The check compares what is on screen with what the server serves over the SAME origin, so a card that
// hard-codes a sentence the server no longer says fails rather than passes. Exit 0 only when they match and
// the page logged no error.
const { chromium } = require('playwright-core');
const [BASE, OUT] = process.argv.slice(2);
const failures = [], errors = [];
const sleep = ms => new Promise(r => setTimeout(r, ms));
function check(cond, msg) { if (cond) console.log('  ok   ' + msg); else { failures.push(msg); console.log('  FAIL ' + msg); } }
const shown = s => (s || '').replace(/\s+/g, ' ').trim();

(async () => {
  const b = await chromium.launch({executablePath: '/ms-playwright/chromium-1229/chrome-linux64/chrome', args: ['--no-sandbox']});
  const page = await b.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', e => errors.push('pageerror: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' <- ')));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  await page.goto(BASE + '/#models');
  await page.waitForSelector('#emap-disclaimer', {timeout: 30000});

  // the card is filled from a request, so wait for it rather than racing it
  let rendered = '';
  for (let i = 0; i < 60; i++) {
    rendered = await page.locator('#emap-disclaimer').innerText().catch(() => '');
    if (shown(rendered)) break;
    await sleep(500);
  }

  const served = await page.evaluate(async () => {
    const r = await fetch('/api/expert-map');
    if (!r.ok) return { error: 'HTTP ' + r.status };
    const j = await r.json();
    return { disclaimer: j.disclaimer || '' };
  }).catch(e => ({ error: e.message }));

  console.log('structure');
  check(await page.locator('#emap .card, #emap').count() >= 1, 'the Expert map card is on the Models page');
  check(!served.error, 'the page could read /api/expert-map: ' + (served.error || 'ok'));
  check(shown(served.disclaimer).length > 0, 'the server serves a disclaimer');
  check(/^This map is a starting guess/.test(shown(served.disclaimer)),
    'the served disclaimer is the expert-map one: ' + shown(served.disclaimer).slice(0, 60) + '...');

  console.log('rendering');
  check(shown(rendered).length > 0, 'the card shows a disclaimer (it was empty before this change)');
  check(shown(rendered) === shown(served.disclaimer), 'the card shows exactly what the server serves');
  const box = await page.locator('#emap-disclaimer').boundingBox();
  check(!!box && box.height > 0 && box.width > 0, 'and it is actually painted (not hidden)');
  check(/does not change the model file/.test(shown(rendered)), 'it carries the promise about the model file');

  await page.locator('#emap').scrollIntoViewIfNeeded();
  await page.screenshot({path: `${OUT}/emap-disclaimer.png`, fullPage: false});

  for (const e of errors) console.log('  page error: ' + e);
  check(errors.length === 0, 'no page errors');
  await b.close();
  console.log(failures.length ? `FAIL ${failures.length}: ${failures.join(' | ')}` : 'PASS: expert-map disclaimer rendered');
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.log('FAIL (harness): ' + (e.stack || e.message)); process.exit(1); });
