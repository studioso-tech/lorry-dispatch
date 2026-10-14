/**
 * 画面に無いカット（半期の体制設計の表・構成図）を静止画にする。
 *
 * 当日の配車の画面には体制設計が無い（掃引の結果なので）。端末の出力をそのまま
 * 映すより読みやすい形で、画面と同じ地味さに合わせたカードにする。
 *
 * 使い方:
 *   node scripts/capture_cards.mjs
 */
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const require = createRequire('C:/Users/studi/Desktop/Greenlight_Studio/');
const { chromium } = require('playwright');

const ROOT = 'C:/Users/studi/Desktop/Lorry_Dispatch';
const OUT = `${ROOT}/media/stills`;
mkdirSync(OUT, { recursive: true });

const run = async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(`${ROOT}/scripts/cards.html`).href, { waitUntil: 'networkidle' });

  for (const [id, name] of [['fleet', 'cut_09_fleet'], ['arch', 'cut_10_arch']]) {
    await page.locator(`#${id}`).screenshot({ path: `${OUT}/${name}.png` });
    console.log('撮影:', name);
  }
  await browser.close();
};

run().catch((e) => { console.error('失敗:', e.message); process.exit(1); });
