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

test('should show error message for invalid search', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await page.locator('input[type="text"]').fill('xyznotapokemon999');
  await page.locator('.search-btn').click();
  await expect(page.locator('.error-msg')).toBeVisible();
  await expect(page.locator('.error-msg')).toContainText('No pokemon found');
});

test('clicking a pokemon card opens detail modal with stats', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  const firstCard = page.locator('.pokemon-card').first();
  const pokemonName = await firstCard.locator('.pokemon-name').textContent();
  await firstCard.click();
  await expect(page.locator('.modal-overlay')).toBeVisible();
  await expect(page.locator('.modal-card')).toBeVisible();
  await expect(page.locator('.modal-body h2')).toContainText(pokemonName!.trim());
  await expect(page.locator('.modal-card .stat-label').first()).toBeVisible();
  await expect(page.locator('.modal-card .close-btn')).toBeVisible();
  await page.locator('.close-btn').click();
  await expect(page.locator('.modal-overlay')).not.toBeVisible();
});

test('should close detail modal by clicking the overlay background', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await page.locator('.pokemon-card').first().click();
  await expect(page.locator('.modal-overlay')).toBeVisible();
  await expect(page.locator('.modal-card')).toBeVisible();
  await page.locator('.modal-overlay').click({ position: { x: 5, y: 5 } });
  await expect(page.locator('.modal-overlay')).not.toBeVisible();
});

test('should display height weight and stat bars in detail modal', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await page.locator('.pokemon-card').first().click();
  await expect(page.locator('.modal-card')).toBeVisible();
  await expect(page.locator('.meta-label').filter({ hasText: 'Height' })).toBeVisible();
  await expect(page.locator('.meta-label').filter({ hasText: 'Weight' })).toBeVisible();
  const heightVal = await page.locator('.meta-item').filter({ hasText: 'Height' }).locator('.meta-value').textContent();
  expect(heightVal).toMatch(/\d.*m/);
  const weightVal = await page.locator('.meta-item').filter({ hasText: 'Weight' }).locator('.meta-value').textContent();
  expect(weightVal).toMatch(/\d.*kg/);
  const statBars = page.locator('.stat-bar');
  await expect(statBars).toHaveCount(4);
  for (let i = 0; i < 4; i++) {
    const width = await statBars.nth(i).evaluate(el => getComputedStyle(el).width);
    expect(parseInt(width)).toBeGreaterThan(0);
  }
});

test('should paginate through pokemon pages', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await expect(page.locator('.page-info')).toContainText('Page 1');
  const prevBtn = page.locator('.page-btn', { hasText: 'Previous' });
  await expect(prevBtn).toBeDisabled();
  const firstPageNames = await page.locator('.pokemon-name').allTextContents();
  await page.locator('.page-btn', { hasText: 'Next' }).click();
  await page.waitForSelector('.pokemon-card', { timeout: 30000 });
  await expect(page.locator('.page-info')).toContainText('Page 2');
  const secondPageNames = await page.locator('.pokemon-name').allTextContents();
  expect(firstPageNames[0]).not.toBe(secondPageNames[0]);
  await expect(prevBtn).not.toBeDisabled();
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