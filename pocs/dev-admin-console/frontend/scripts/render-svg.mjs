import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
const svg = readFileSync(process.argv[2], 'utf8');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 });
await page.setContent(`<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap" rel="stylesheet">
<style>html,body{margin:0;padding:0;background:#FAF5EE}</style></head><body>${svg}</body></html>`,
  { waitUntil: 'networkidle' });
await page.evaluate(() => document.fonts.ready);
await new Promise(r => setTimeout(r, 1200));
await page.screenshot({ path: process.argv[3], clip: { x: 0, y: 0, width: 1280, height: 860 } });
await browser.close();
console.log('rendered ' + process.argv[3]);
