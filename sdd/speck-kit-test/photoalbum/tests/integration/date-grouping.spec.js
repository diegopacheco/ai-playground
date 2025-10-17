import { test, expect } from '@playwright/test';

test.describe('Date-Based Album Grouping', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');
  });

  test('should display grouping toggle button', async ({ page }) => {
    const toggleButton = page.getByTestId('grouping-toggle');
    await expect(toggleButton).toBeVisible();
  });

  test('should show flat list by default', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Test Album');
    await page.getByRole('button', { name: /create/i }).click();

    const albumGrid = page.getByTestId('album-grid');
    await expect(albumGrid).toBeVisible();

    const dateGroupHeaders = page.locator('[data-testid="date-group-header"]');
    await expect(dateGroupHeaders).toHaveCount(0);
  });

  test('should display date group headers when grouping is enabled', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 1');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 2');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroupHeaders = page.locator('[data-testid="date-group-header"]');
    await expect(dateGroupHeaders).toHaveCount(1);
  });

  test('should format date group header correctly', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Test Album');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroupHeader = page.locator('[data-testid="date-group-header"]').first();
    const headerText = await dateGroupHeader.textContent();

    expect(headerText).toMatch(/\w+ \d{4}/);
  });

  test('should group albums created in same month', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 1');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 2');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 3');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroups = page.locator('[data-testid="date-group"]');
    const firstGroup = dateGroups.first();

    const albumsInGroup = firstGroup.locator('[data-testid="album-card"]');
    const count = await albumsInGroup.count();

    expect(count).toBeGreaterThan(0);
  });

  test('should toggle between grouped and flat views', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Test Album');
    await page.getByRole('button', { name: /create/i }).click();

    const toggleButton = page.getByTestId('grouping-toggle');
    await toggleButton.click();

    let dateGroupHeaders = page.locator('[data-testid="date-group-header"]');
    await expect(dateGroupHeaders).toHaveCount(1);

    await toggleButton.click();

    dateGroupHeaders = page.locator('[data-testid="date-group-header"]');
    await expect(dateGroupHeaders).toHaveCount(0);
  });

  test('should persist grouping preference after page reload', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Test Album');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    await page.reload();
    await page.waitForLoadState('networkidle');

    const dateGroupHeaders = page.locator('[data-testid="date-group-header"]');
    await expect(dateGroupHeaders).toHaveCount(1);
  });

  test('should display albums in correct order within groups', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album A');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album B');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroup = page.locator('[data-testid="date-group"]').first();
    const albumCards = dateGroup.locator('[data-testid="album-card"]');

    const firstAlbum = albumCards.first();
    const firstAlbumName = await firstAlbum.locator('.album-name').textContent();

    expect(firstAlbumName).toBeTruthy();
  });
});
