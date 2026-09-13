import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const build = join(here, 'build');
const iconset = join(build, 'icon.iconset');
const logo = readFileSync(join(here, '..', 'public', 'logo.svg'), 'utf8');

rmSync(build, { recursive: true, force: true });
mkdirSync(iconset, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1024, height: 1024 } });
await page.setContent(`<body style="margin:0;width:1024px;height:1024px;display:grid;place-items:center;background:transparent"><div style="width:824px;height:824px;border-radius:186px;background:#214c40;box-shadow:0 18px 40px #0003">${logo.replace(/<rect[^>]*\/>/, '').replace('<svg ', '<svg width="824" height="824" ')}</div></body>`);
await page.screenshot({ path: join(build, 'icon-1024.png'), omitBackground: true });
await browser.close();
for (const size of [16, 32, 64, 128, 256, 512]) {
  execFileSync('sips', ['-z', String(size), String(size), join(build, 'icon-1024.png'), '--out', join(iconset, `icon_${size}x${size}.png`)], { stdio: 'ignore' });
  execFileSync('sips', ['-z', String(size * 2), String(size * 2), join(build, 'icon-1024.png'), '--out', join(iconset, `icon_${size}x${size}@2x.png`)], { stdio: 'ignore' });
}
execFileSync('iconutil', ['-c', 'icns', iconset, '-o', join(build, 'icon.icns')]);
console.log(`Icon written to ${join(build, 'icon.icns')}`);
