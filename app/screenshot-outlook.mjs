// Screenshot-verify the /outlook page against the REAL artifact (anti-"basura" rule).
// Usage: node screenshot-outlook.mjs  (expects `vite preview` running, or pass BASE=...)
import { chromium } from 'playwright';

const base = (process.env.BASE || 'http://localhost:4173').replace(/\/+$/, '');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 2000 }, deviceScaleFactor: 1 });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push('PAGEERROR: ' + String(e)));
page.on('requestfailed', (r) => { const u = r.url(); if (/data\/outlook/.test(u)) errors.push('REQFAIL: ' + u); });

try {
  await page.goto(`${base}/outlook`, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(5000); // let OSM tiles + the deck.gl heatmap paint
  // surface what rendered
  const h1 = await page.locator('h1').first().textContent().catch(() => null);
  const statNums = await page.locator('.outlook-stats .stat-num').allTextContents().catch(() => []);
  const evRows = await page.locator('.bench-table tbody tr').count().catch(() => 0);
  const zones = await page.locator('.outlook-zones li').count().catch(() => 0);
  const hasCanvas = (await page.locator('.map-canvas canvas').count().catch(() => 0)) > 0;
  await page.screenshot({ path: 'outlook-shot.png', fullPage: true });
  console.log(JSON.stringify({ h1, statNums, evRows, zones, hasCanvas, errors }, null, 2));
} catch (e) {
  console.log('NAVIGATION FAILED:', String(e), '| errors:', errors);
} finally {
  await browser.close();
}
