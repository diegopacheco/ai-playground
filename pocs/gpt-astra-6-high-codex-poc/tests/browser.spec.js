import { test, expect } from '@playwright/test';

test('all locations render, riders switch, and instructions open', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.evaluate(() => document.fonts.ready);
  await expect(page.locator('#game')).toHaveAttribute('data-status', 'ready');
  await page.screenshot({ path: 'printscreens/embarcadero.png', fullPage: true });
  await page.locator('[data-rider="girl"]').click();
  await expect(page.locator('[data-rider="girl"]')).toHaveAttribute('aria-pressed', 'true');
  for (const spot of ['wharf', 'ferry', 'pier39']) {
    await page.locator(`[data-spot="${spot}"]`).click();
    await expect(page.locator(`[data-spot="${spot}"]`)).toHaveAttribute('aria-pressed', 'true');
    await page.screenshot({ path: `printscreens/${spot}.png`, fullPage: true });
  }
  await page.locator('#guide-open').click();
  await expect(page.locator('#guide')).toBeVisible();
  await page.screenshot({ path: 'printscreens/instructions.png', fullPage: true });
  await page.locator('#guide-ready').click();
  await expect(page.locator('#guide')).not.toBeVisible();
  expect(errors).toEqual([]);
});

test('keyboard tricks bank points, pause freezes time, and best survives reload', async ({ page }) => {
  await page.clock.install();
  await page.goto('/');
  await page.locator('#start').click();
  await expect(page.locator('#game')).toHaveAttribute('data-status', 'playing');
  await page.keyboard.press('Space');
  await page.clock.runFor(200);
  await page.keyboard.press('j');
  await page.keyboard.press('k');
  await page.keyboard.press('l');
  await page.clock.runFor(850);
  await expect(page.locator('#score')).toHaveText('01950');
  await page.screenshot({ path: 'printscreens/playing.png', fullPage: true });
  await page.keyboard.press('p');
  await expect(page.locator('#game')).toHaveAttribute('data-status', 'paused');
  const time = await page.locator('#time').textContent();
  await page.clock.runFor(2500);
  await expect(page.locator('#time')).toHaveText(time);
  await page.locator('#start').click();
  await expect(page.locator('#game')).toHaveAttribute('data-status', 'playing');
  await page.reload();
  await expect(page.locator('#best')).toContainText('1,950');
});

test('a full run finishes and can restart', async ({ page }) => {
  await page.clock.install();
  await page.goto('/');
  await page.locator('#start').click();
  await page.clock.runFor(61000);
  await expect(page.locator('#game')).toHaveAttribute('data-status', 'finished');
  await expect(page.locator('#time')).toHaveText('00:00');
  await expect(page.locator('#start')).toContainText('Ride again');
  await page.locator('#start').click();
  await expect(page.locator('#game')).toHaveAttribute('data-status', 'playing');
  await expect(page.locator('#time')).toHaveText('01:00');
});

test('mobile fits the screen and touch buttons perform tricks', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.clock.install();
  await page.goto('/');
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'printscreens/mobile.png', fullPage: true });
  await page.locator('#start').click();
  await page.locator('[data-action="jump"]').click();
  await page.clock.runFor(200);
  await page.locator('[data-action="kickflip"]').click();
  await page.clock.runFor(850);
  await expect(page.locator('#score')).toHaveText('00200');
});
