import { test, expect } from '@playwright/test';

test.describe('Create Album Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');
  });

  test('should display create album button on main page', async ({ page }) => {
    const createButton = page.locator('button:has-text("Create Album")');
    await expect(createButton).toBeVisible();
  });

  test('should open modal when create album button is clicked', async ({ page }) => {
    await page.click('button:has-text("Create Album")');

    const modal = page.locator('[data-testid="create-album-modal"]');
    await expect(modal).toBeVisible();
  });

  test('should create new album with valid name', async ({ page }) => {
    await page.click('button:has-text("Create Album")');

    await page.fill('[data-testid="album-name-input"]', 'My Vacation Photos');
    await page.click('[data-testid="create-album-submit"]');

    const albumCard = page.locator('[data-testid="album-card"]:has-text("My Vacation Photos")');
    await expect(albumCard).toBeVisible();
  });

  test('should close modal after successful album creation', async ({ page }) => {
    await page.click('button:has-text("Create Album")');

    await page.fill('[data-testid="album-name-input"]', 'Summer 2025');
    await page.click('[data-testid="create-album-submit"]');

    const modal = page.locator('[data-testid="create-album-modal"]');
    await expect(modal).not.toBeVisible();
  });

  test('should show validation error for empty album name', async ({ page }) => {
    await page.click('button:has-text("Create Album")');

    await page.click('[data-testid="create-album-submit"]');

    const error = page.locator('[data-testid="album-name-error"]');
    await expect(error).toBeVisible();
    await expect(error).toContainText('name');
  });

  test('should cancel album creation when cancel button is clicked', async ({ page }) => {
    await page.click('button:has-text("Create Album")');

    await page.fill('[data-testid="album-name-input"]', 'Test Album');
    await page.click('[data-testid="create-album-cancel"]');

    const modal = page.locator('[data-testid="create-album-modal"]');
    await expect(modal).not.toBeVisible();

    const albumCard = page.locator('[data-testid="album-card"]:has-text("Test Album")');
    await expect(albumCard).not.toBeVisible();
  });

  test('should display newly created album in grid layout', async ({ page }) => {
    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Grid Test Album');
    await page.click('[data-testid="create-album-submit"]');

    const grid = page.locator('[data-testid="album-grid"]');
    await expect(grid).toBeVisible();

    const albumCard = grid.locator('[data-testid="album-card"]:has-text("Grid Test Album")');
    await expect(albumCard).toBeVisible();
  });

  test('should show album creation date on album card', async ({ page }) => {
    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Dated Album');
    await page.click('[data-testid="create-album-submit"]');

    const albumCard = page.locator('[data-testid="album-card"]:has-text("Dated Album")');
    const dateElement = albumCard.locator('[data-testid="album-date"]');
    await expect(dateElement).toBeVisible();
  });
});
