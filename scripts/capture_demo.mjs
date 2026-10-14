/**
 * デモ動画の静止画を、**公開中の本番画面**から撮る。
 *
 * 画面の見え方そのものを素材にする（作り直した模型ではない）。求め方は
 * 「再現」に固定する。即応だと撮り直すたびに担当も時刻も変わり、ナレーションの
 * 数字と画面が食い違うため（docs/デモ筋書き.md）。
 *
 * playwright は Greenlight_Studio の node_modules を借りる（この枝に npm の
 * 依存を増やさないため）。
 *
 * 使い方:
 *   NODE_PATH="C:/Users/studi/Desktop/Greenlight_Studio/node_modules" \
 *     node scripts/capture_demo.mjs
 */
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';

// playwright は Greenlight_Studio の node_modules から読む（ESM の bare import は
// NODE_PATH を見ないので、解決の起点を明示する）
const require = createRequire('C:/Users/studi/Desktop/Greenlight_Studio/');
const { chromium } = require('playwright');

const URL = 'https://lorry-dispatch-125474373169.us-central1.run.app/?k=marifu-lorry-2026';
const OUT = 'C:/Users/studi/Desktop/Lorry_Dispatch/media/stills';
// **画面の文字が動画で読める大きさになるように撮る。**
// 画面の本体は幅1080pxまでなので、1920pxで撮ると右半分が余白になり、文字が
// 小さいまま縮む。狭い幅（1280）で撮り、端末の倍率を1.5倍にして実寸1920px
// にする。こうすると本体が画面いっぱいに写り、文字も大きい。
const W = 1280, H = 760, SCALE = 1.5;
const SOLVE_TIMEOUT = 240_000;   // 再現モードの求解は本番で約80秒

mkdirSync(OUT, { recursive: true });

const shot = async (page, name, opts = {}) => {
  await page.screenshot({ path: `${OUT}/${name}.png`, ...opts });
  console.log('撮影:', name);
};

const run = async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: SCALE });
  page.on('console', (m) => { if (m.type() === 'error') console.log('画面のエラー:', m.text()); });

  // --- 幕1: 朝の計画 -------------------------------------------------------
  await page.goto(URL, { waitUntil: 'networkidle' });
  await shot(page, 'cut_01_top');

  await page.selectOption('[name=mode]', 'reproducible');
  await page.click('#plan-form .go');
  await page.waitForSelector('#plan-summary:not([hidden])', { timeout: SOLVE_TIMEOUT });
  await page.waitForTimeout(500);
  await shot(page, 'cut_02_plan');

  // 6号車の札（宇品港SSを運ぶ車）を大きく見せる
  const sixth = page.locator('.vehicle', { hasText: '6号車' }).first();
  if (await sixth.count()) {
    await sixth.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    await shot(page, 'cut_03_vehicle');
  } else {
    await page.mouse.wheel(0, 700);
    await page.waitForTimeout(300);
    await shot(page, 'cut_03_vehicle');
  }

  // --- 幕2・幕3: FAX → 組み直し（事象を当てない） --------------------------
  await page.click('.tab[data-panel=intake]');
  await page.waitForTimeout(300);
  await shot(page, 'cut_04_intake_top');

  // **折りたたみを先に開く。** 閉じたままだと中のチェックに触れず、
  // 「当てない」側が撮れない（最初の録りでそうなった）
  await page.locator('details.event summary').click();
  await page.waitForTimeout(200);
  await page.uncheck('[name=ev-on]');
  await page.click('#intake-form .go');
  await page.waitForSelector('#intake-result:not([hidden])', { timeout: SOLVE_TIMEOUT });
  await page.waitForTimeout(500);
  await shot(page, 'cut_05_read');
  await page.locator('#intake-result').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await shot(page, 'cut_06_shortfall');

  // --- 幕3後半: 事象を当てる ----------------------------------------------
  await page.check('[name=ev-on]');
  await page.locator('details.event').scrollIntoViewIfNeeded();
  await shot(page, 'cut_07a_event');
  await page.click('#intake-form .go');
  await page.waitForSelector('#intake-result:not([hidden])', { timeout: SOLVE_TIMEOUT });
  await page.waitForTimeout(500);
  await page.locator('#intake-result').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await shot(page, 'cut_07_filled');

  // --- 幕4: 持ち戻り -------------------------------------------------------
  await page.click('.tab[data-panel=ret]');
  await page.waitForTimeout(300);
  const stop = page.locator('[name=stop]');
  const value = await stop.locator('option', { hasText: '宇品港' }).first().getAttribute('value');
  if (value) await stop.selectOption(value);
  await page.click('#ret-form .go');
  await page.waitForSelector('#ret-result:not([hidden])', { timeout: SOLVE_TIMEOUT });
  await page.waitForTimeout(500);
  await shot(page, 'cut_08_return');
  await page.locator('#ret-result table').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await shot(page, 'cut_08b_options');

  await browser.close();
  console.log('完了:', OUT);
};

run().catch((e) => { console.error('失敗:', e.message); process.exit(1); });
