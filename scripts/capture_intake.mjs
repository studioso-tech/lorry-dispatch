/**
 * 幕2・幕3だけ撮り直す。
 *
 * **折りたたみ（details）は先に開く。** 閉じたままだと中のチェックボックスに
 * 触れず、「事象を当てない」側が撮れない（最初の録りで、当てた結果が
 * 当てない場面として撮れていた）。
 *
 * 使い方:
 *   node scripts/capture_intake.mjs
 */
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';

const require = createRequire('C:/Users/studi/Desktop/Greenlight_Studio/');
const { chromium } = require('playwright');

const URL = 'https://lorry-dispatch-125474373169.us-central1.run.app/?k=marifu-lorry-2026';
const OUT = 'C:/Users/studi/Desktop/Lorry_Dispatch/media/stills';
const SOLVE_TIMEOUT = 240_000;

mkdirSync(OUT, { recursive: true });

const run = async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto(URL, { waitUntil: 'networkidle' });

  // 画面の「求め方」を再現にそろえる（差し込みはこの条件をそのまま送る）
  await page.selectOption('[name=mode]', 'reproducible');
  await page.click('.tab[data-panel=intake]');
  await page.waitForTimeout(300);

  // **折りたたみを開いてから**チェックを外す
  await page.locator('details.event summary').click();
  await page.waitForTimeout(200);
  await page.uncheck('[name=ev-on]');
  console.log('事象: 当てない');

  await page.click('#intake-form .go');
  await page.waitForSelector('#intake-result:not([hidden])', { timeout: SOLVE_TIMEOUT });
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}/cut_05_read.png` });
  await page.locator('#intake-result').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/cut_06_shortfall.png` });
  console.log('撮影: cut_05_read / cut_06_shortfall');

  // 事象を当てる
  await page.check('[name=ev-on]');
  await page.click('#intake-form .go');
  await page.waitForSelector('#intake-result:not([hidden])', { timeout: SOLVE_TIMEOUT });
  await page.waitForTimeout(600);
  await page.locator('#intake-result').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/cut_07_filled.png` });
  console.log('撮影: cut_07_filled');

  await browser.close();
  console.log('完了');
};

run().catch((e) => { console.error('失敗:', e.message); process.exit(1); });
