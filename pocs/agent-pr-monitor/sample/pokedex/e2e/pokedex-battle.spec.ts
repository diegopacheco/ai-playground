import { test, expect } from '@playwright/test';

test('pokedex tab loads and displays pokemon cards', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toHaveText('Pokedex');
  await expect(page.locator('.tab-btn.active')).toHaveText('Pokedex');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  const cards = page.locator('.pokemon-card');
  await expect(cards).not.toHaveCount(0);
  const firstName = await cards.first().locator('.pokemon-name').textContent();
  expect(firstName).toBeTruthy();
});

test('clicking a pokemon card opens detail modal with stats', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await page.locator('.pokemon-card').first().click();
  await expect(page.locator('.modal-overlay')).toBeVisible();
  await expect(page.locator('.modal-card .modal-img')).toBeVisible();
  await expect(page.locator('.stat-label')).toHaveCount(4);
  const statLabels = await page.locator('.stat-label').allTextContents();
  expect(statLabels).toEqual(['HP', 'ATK', 'DEF', 'SPD']);
  await page.locator('.close-btn').click();
  await expect(page.locator('.modal-overlay')).not.toBeVisible();
});

test('search filters pokemon by name', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await page.locator('.search-bar input').fill('pikachu');
  await page.locator('.search-btn').click();
  await page.waitForSelector('.pokemon-card', { timeout: 15000 });
  const cards = page.locator('.pokemon-card');
  await expect(cards).toHaveCount(1);
  await expect(cards.first().locator('.pokemon-name')).toHaveText('Pikachu');
});

test('battle tab allows selecting two pokemon and shows VS badge', async ({ page }) => {
  await page.goto('/');
  await page.locator('.tab-btn', { hasText: 'Battle' }).click();
  await expect(page.locator('.selection-title')).toContainText('Choose YOUR Pokemon');
  await page.waitForSelector('.pick-card', { timeout: 30000 });
  await page.locator('.pick-card').first().click();
  await expect(page.locator('.selection-title')).toContainText('Choose OPPONENT Pokemon');
  await expect(page.locator('.vs-badge')).toHaveText('VS');
  await page.locator('.pick-card').nth(1).click();
  await expect(page.locator('.selection-title')).toContainText('Ready to Battle!');
  await expect(page.locator('.fight-btn')).toBeVisible();
});

test('full battle flow: start battle, attack until victory, then reset', async ({ page }) => {
  await page.goto('/');
  await page.locator('.tab-btn', { hasText: 'Battle' }).click();
  await page.waitForSelector('.pick-card', { timeout: 30000 });
  await page.locator('.pick-card').first().click();
  await page.locator('.pick-card').nth(1).click();
  await page.locator('.fight-btn').click();
  await expect(page.locator('.arena')).toBeVisible();
  await expect(page.locator('.attack-btn')).toBeVisible();
  const playerName = await page.locator('.player-side .fighter-name').textContent();
  const enemyName = await page.locator('.enemy-side .fighter-name').textContent();
  expect(playerName).toBeTruthy();
  expect(enemyName).toBeTruthy();
  for (let i = 0; i < 100; i++) {
    const attackBtn = page.locator('.attack-btn');
    if (await attackBtn.isVisible().catch(() => false)) {
      await attackBtn.click();
    } else {
      break;
    }
  }
  await expect(page.locator('.log-victory')).toBeVisible({ timeout: 5000 });
  const victoryText = await page.locator('.log-victory').last().textContent();
  expect(victoryText).toContain('wins!');
  await expect(page.locator('.reset-btn')).toBeVisible();
  await page.locator('.reset-btn').click();
  await expect(page.locator('.selection-title')).toContainText('Choose YOUR Pokemon');
});