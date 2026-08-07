// UI tests for the Chart tab legend interactions (hover, single click,
// double click) in web/scores.js. They drive a real headless browser via
// puppeteer, serving the site files locally (Chart.js is a vendored copy in
// vendor/chart.umd.js).
//
// Run only these tests with: make test-ui
// Run the whole suite with:  make test
'use strict';

const { test, before, after } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '..', '..');
const WEB = path.join(ROOT, 'web');
const DATA = path.join(ROOT, 'data', 'models-prediction.json');
const CHART_UMD = path.join(ROOT, 'vendor', 'chart.umd.js');

let puppeteer;
try {
  puppeteer = require('puppeteer');
} catch (err) {
  puppeteer = null;
}

const load = p => fs.readFileSync(p, 'utf8');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

let browser;
let page;

before(async () => {
  if (!puppeteer) return;
  browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 950 });

  await page.setRequestInterception(true);
  page.on('request', req => {
    const url = req.url();
    if (url.endsWith('/index.html')) {
      req.respond({ status: 200, contentType: 'text/html', body: load(path.join(WEB, 'index.html')) });
    } else if (url.endsWith('storage.js')) {
      req.respond({ status: 200, contentType: 'text/javascript', body: load(path.join(WEB, 'storage.js')) });
    } else if (url.endsWith('scores.js')) {
      req.respond({ status: 200, contentType: 'text/javascript', body: load(path.join(WEB, 'scores.js')) });
    } else if (url.endsWith('chart.js') || url.includes('chart.umd')) {
      req.respond({ status: 200, contentType: 'text/javascript', body: load(CHART_UMD) });
    } else if (url.endsWith('models-prediction.json')) {
      req.respond({ status: 200, contentType: 'application/json', body: load(DATA) });
    } else {
      req.respond({ status: 404, contentType: 'text/html', body: 'not found' });
    }
  });
});

after(async () => {
  if (browser) await browser.close();
});

const skipOpts = puppeteer
  ? {}
  : { skip: 'puppeteer not installed (run `npm install` from the repository root)' };

// Load the page fresh so each test starts from a clean state.
async function loadPage() {
  await page.goto('http://localhost/index.html', { waitUntil: 'networkidle2', timeout: 30000 });
  await sleep(300);
}

// Compute client coordinates of a legend item's hitbox.
async function legendPosition(index) {
  return page.evaluate(idx => {
    const canvas = document.getElementById('chart-canvas');
    const chart = window.Chart.getChart(canvas);
    const legend = chart.legend;
    const rect = canvas.getBoundingClientRect();
    const hitbox = legend.legendHitBoxes[idx];
    return {
      x: rect.left + (hitbox.left + hitbox.width / 2) * (rect.width / canvas.width),
      y: rect.top + (hitbox.top + hitbox.height / 2) * (rect.height / canvas.height),
      company: legend.legendItems[idx].text
    };
  }, index);
}

// Companies currently visible on the chart.
async function visibleCompanies() {
  return page.evaluate(() => {
    const canvas = document.getElementById('chart-canvas');
    const chart = window.Chart.getChart(canvas);
    return [...new Set(chart.data.datasets.filter(d => !d.hidden).map(d => d.data[0].company))];
  });
}

test('chart tab legend interactions', skipOpts, async t => {
  await t.test('handlers are registered at the legend level', async () => {
    await loadPage();
    const info = await page.evaluate(() => {
      const canvas = document.getElementById('chart-canvas');
      const chart = window.Chart.getChart(canvas);
      const legend = chart.legend;
      return {
        onClick: typeof legend.options.onClick,
        onHover: typeof legend.options.onHover,
        onLeave: typeof legend.options.onLeave,
        labelsOnClick: typeof (legend.options.labels || {}).onClick
      };
    });
    assert.deepStrictEqual(
      { onClick: info.onClick, onHover: info.onHover, onLeave: info.onLeave },
      { onClick: 'function', onHover: 'function', onLeave: 'function' }
    );
    assert.strictEqual(info.labelsOnClick, 'undefined');
  });

  await t.test('hover isolates the hovered company and restores on leave', async () => {
    await loadPage();
    const pos = await legendPosition(1);
    await page.mouse.move(pos.x, pos.y);
    await sleep(300);
    const during = await visibleCompanies();
    assert.deepStrictEqual(during, [pos.company], 'only the hovered company should be visible');

    await page.mouse.move(5, 5);
    await sleep(300);
    const count = await page.evaluate(() => {
      const chart = window.Chart.getChart(document.getElementById('chart-canvas'));
      return chart.data.datasets.filter(d => !d.hidden).length;
    });
    assert.strictEqual(count, await page.evaluate(() => {
      const chart = window.Chart.getChart(document.getElementById('chart-canvas'));
      return chart.data.datasets.length;
    }), 'all datasets should be visible again after leaving');
  });

  await t.test('single click toggles a company on and off', async () => {
    await loadPage();
    const pos = await legendPosition(0);
    // Click once: the company disappears.
    await page.mouse.click(pos.x, pos.y, { count: 1 });
    await sleep(300);
    let visible = await visibleCompanies();
    assert.ok(!visible.includes(pos.company), 'company should be hidden after one click');

    // Click again: the company reappears.
    await page.mouse.click(pos.x, pos.y, { count: 1 });
    await sleep(300);
    visible = await visibleCompanies();
    assert.ok(visible.includes(pos.company), 'company should be visible after a second click');
  });

  await t.test('double click hides every other company and shows only the clicked one', async () => {
    await loadPage();
    const pos = await legendPosition(2);
    await page.mouse.click(pos.x, pos.y, { count: 2 });
    await sleep(300);
    const visible = await visibleCompanies();
    assert.deepStrictEqual(visible, [pos.company], 'only the double-clicked company should be visible');
  });
});
