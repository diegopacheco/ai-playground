import { test, expect } from '@playwright/test';

test('pokedex loads and displays pokemon cards', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  const cards = page.locator('.pokemon-card');
  await expect(cards.first()).toBeVisible();
  const count = await cards.count();
  expect(count).toBeGreaterThanOrEqual(1);
  await expect(page.locator('.pokemon-name').first()).not.toBeEmpty();
  await expect(page.locator('.pokemon-id').first()).toContainText('#');
});

test('search pokemon by name returns result', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  const searchInput = page.locator('.search-bar input');
  await searchInput.fill('pikachu');
  await page.locator('.search-btn').click();
  await page.waitForSelector('.pokemon-card', { timeout: 15000 });
  const cards = page.locator('.pokemon-card');
  await expect(cards).toHaveCount(1);
  await expect(page.locator('.pokemon-name').first()).toContainText('Pikachu');
});

test('clicking a pokemon card opens detail modal with stats', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await page.locator('.pokemon-card').first().click();
  await expect(page.locator('.modal-overlay')).toBeVisible();
  await expect(page.locator('.modal-card')).toBeVisible();
  await expect(page.locator('.modal-card .stat-label').first()).toBeVisible();
  await expect(page.locator('.modal-card .close-btn')).toBeVisible();
  await page.locator('.close-btn').click();
  await expect(page.locator('.modal-overlay')).not.toBeVisible();
});

test('battle tab allows selecting two pokemon and starting a battle', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.tab-btn', { timeout: 30000 });
  await page.locator('.tab-btn', { hasText: 'Battle' }).click();
  await expect(page.locator('.selection-title')).toContainText('Choose YOUR Pokemon');
  await page.waitForSelector('.pick-card', { timeout: 30000 });
  await page.locator('.pick-card').first().click();
  await expect(page.locator('.selection-title')).toContainText('Choose OPPONENT Pokemon');
  await page.locator('.pick-card').nth(1).click();
  await expect(page.locator('.fight-btn')).toBeVisible();
  const chosenNames = page.locator('.chosen-name');
  await expect(chosenNames).toHaveCount(2);
});

test('battle executes attacks and shows battle log until victory', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.tab-btn', { timeout: 30000 });
  await page.locator('.tab-btn', { hasText: 'Battle' }).click();
  await page.waitForSelector('.pick-card', { timeout: 30000 });
  await page.locator('.pick-card').first().click();
  await page.locator('.pick-card').nth(1).click();
  await page.locator('.fight-btn').click();
  await expect(page.locator('.arena')).toBeVisible();
  await expect(page.locator('.log-entry').first()).toContainText('FIGHT!');
  while (await page.locator('.attack-btn').isVisible()) {
    await page.locator('.attack-btn').click();
    await page.waitForTimeout(100);
  }
  await expect(page.locator('.log-entry.log-victory')).toBeVisible();
  const victoryText = await page.locator('.log-entry.log-victory').textContent();
  expect(victoryText).toContain('wins!');
  await expect(page.locator('.reset-btn')).toBeVisible();
  await page.locator('.reset-btn').click();
  await expect(page.locator('.selection-title')).toContainText('Choose YOUR Pokemon');
});