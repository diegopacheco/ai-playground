const { test, expect } = require('@playwright/test');
test('each layer exposes its engineering details without leaving the viewer', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.layer')).toHaveCount(6);
  const names = ['Sapphire crystal', 'AMOLED display', 'Titanium housing', 'Logic & power', 'Sensor array', 'Sport band'];
  for (let index = 0; index < names.length; index++) {
    await page.locator('.layer').nth(index).click();
    await expect(page.locator('#detail-title')).toHaveText(names[index]);
    await expect(page.locator('.layer[aria-pressed="true"]')).toHaveCount(1);
    await expect(page.locator('#detail-description')).not.toBeEmpty();
  }
  expect(errors).toEqual([]);
});
test('assembly, separation, and finish controls change the rendered watch', async ({ page }) => {
  await page.goto('/');
  const scene = page.locator('#scene');
  const initial = await scene.screenshot();
  await page.getByRole('button', { name: 'Assembled', exact: true }).click();
  await expect(page.locator('#separation-value')).toHaveText('0%');
  await expect(page.locator('#view-title')).toHaveText('ASSEMBLED WATCH');
  await page.waitForFunction(() => separation === 0);
  expect((await scene.screenshot()).equals(initial)).toBe(false);
  await page.getByRole('button', { name: 'Graphite titanium', exact: true }).click();
  await expect(page.locator('#finish-name')).toHaveText('Graphite titanium');
  const graphite = await scene.screenshot();
  await page.getByRole('button', { name: 'Natural titanium', exact: true }).click();
  expect((await scene.screenshot()).equals(graphite)).toBe(false);
  await page.locator('#separation').fill('45');
  await expect(page.locator('#separation-value')).toHaveText('45%');
  await expect(page.getByRole('button', { name: 'Exploded', exact: true })).toHaveAttribute('aria-pressed', 'true');
});
test('orbit, zoom, annotations, auto rotation, and reset work', async ({ page }) => {
  await page.goto('/');
  const scene = page.locator('#scene');
  const initial = await scene.screenshot();
  await scene.focus();
  await page.keyboard.press('ArrowRight');
  expect((await scene.screenshot()).equals(initial)).toBe(false);
  const box = await scene.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 80, box.y + box.height / 2 + 20, { steps: 8 });
  await page.mouse.up();
  expect(await page.evaluate(() => yaw)).toBeGreaterThan(-.3);
  await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
  expect(await page.evaluate(() => zoom)).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'Toggle annotations' }).click();
  await expect(page.getByRole('button', { name: 'Toggle annotations' })).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'Auto rotate', exact: true }).click();
  const before = await page.evaluate(() => yaw);
  await page.waitForFunction(value => yaw > value + .01, before);
  await page.getByRole('button', { name: 'Reset view' }).click();
  await expect(page.getByRole('button', { name: 'Auto rotate', exact: true })).toHaveAttribute('aria-pressed', 'false');
  expect(await page.evaluate(() => ({ yaw, pitch, zoom }))).toEqual({ yaw: -.48, pitch: .88, zoom: 1 });
});
test('export downloads a real PNG', async ({ page }) => {
  await page.goto('/');
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export view' }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('astra-one-schematic.png');
  const fs = require('node:fs');
  const buffer = fs.readFileSync(await download.path());
  expect(buffer.subarray(1, 4).toString()).toBe('PNG');
  expect(buffer.length).toBeGreaterThan(20000);
});
test('desktop and mobile remain usable and capture all assembly views', async ({ page }) => {
  await page.goto('/');
  await page.screenshot({ path: 'printscreens/exploded-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Assembled', exact: true }).click();
  await page.waitForFunction(() => separation === 0);
  await page.screenshot({ path: 'printscreens/assembled-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Exploded', exact: true }).click();
  await page.locator('.layer').nth(3).click();
  await page.waitForFunction(() => separation === .75);
  await page.screenshot({ path: 'printscreens/logic-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await expect(page.locator('#detail-title')).toHaveText('Logic & power');
  await page.screenshot({ path: 'printscreens/mobile.png', fullPage: true });
});
test('the server keeps project and operational files private', async ({ request }) => {
  for (const route of ['/scripts/ports.env', '/package.json', '/server.mjs', '/.run/frontend.pid', '/%2e%2e/package.json']) {
    expect((await request.get(route)).status()).toBe(404);
  }
});
