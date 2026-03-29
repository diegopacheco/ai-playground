import { test, expect } from '@playwright/test';

test('should load the pokedex and display pokemon cards', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-grid');
  const cards = page.locator('.pokemon-card');
  await expect(cards.first()).toBeVisible();
  const count = await cards.count();
  expect(count).toBeGreaterThanOrEqual(1);
  await expect(page.locator('.pokemon-name').first()).not.toBeEmpty();
  await expect(page.locator('.pokemon-id').first()).toContainText('#');
});

test('should open pokemon detail modal when clicking a card', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card');
  const firstCard = page.locator('.pokemon-card').first();
  const pokemonName = await firstCard.locator('.pokemon-name').textContent();
  await firstCard.click();
  await expect(page.locator('.modal-overlay')).toBeVisible();
  await expect(page.locator('.modal-card')).toBeVisible();
  await expect(page.locator('.modal-body h2')).toContainText(pokemonName!.trim());
  await expect(page.locator('.stat-label')).toHaveCount(4);
  await page.locator('.close-btn').click();
  await expect(page.locator('.modal-overlay')).not.toBeVisible();
});

test('should search for a pokemon by name and show results', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-grid');
  await page.locator('input[type="text"]').fill('pikachu');
  await page.locator('.search-btn').click();
  await page.waitForSelector('.pokemon-card');
  const cards = page.locator('.pokemon-card');
  await expect(cards).toHaveCount(1);
  await expect(page.locator('.pokemon-name').first()).toContainText('Pikachu');
});

test('should show error message for invalid search', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-grid');
  await page.locator('input[type="text"]').fill('xyznotapokemon999');
  await page.locator('.search-btn').click();
  await expect(page.locator('.error-msg')).toBeVisible();
  await expect(page.locator('.error-msg')).toContainText('No pokemon found');
});

test('should paginate through pokemon pages', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-grid');
  await expect(page.locator('.page-info')).toContainText('Page 1');
  const prevBtn = page.locator('.page-btn', { hasText: 'Previous' });
  await expect(prevBtn).toBeDisabled();
  const firstPageNames = await page.locator('.pokemon-name').allTextContents();
  await page.locator('.page-btn', { hasText: 'Next' }).click();
  await page.waitForSelector('.pokemon-grid');
  await expect(page.locator('.page-info')).toContainText('Page 2');
  const secondPageNames = await page.locator('.pokemon-name').allTextContents();
  expect(firstPageNames[0]).not.toBe(secondPageNames[0]);
  await expect(prevBtn).not.toBeDisabled();
});

test('should close detail modal by clicking the overlay background', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card');
  await page.locator('.pokemon-card').first().click();
  await expect(page.locator('.modal-overlay')).toBeVisible();
  await expect(page.locator('.modal-card')).toBeVisible();
  await page.locator('.modal-overlay').click({ position: { x: 5, y: 5 } });
  await expect(page.locator('.modal-overlay')).not.toBeVisible();
});

test('should display height weight and stat bars in detail modal', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.pokemon-card');
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
