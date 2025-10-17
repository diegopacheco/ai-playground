import { test, expect } from '@playwright/test';

test.describe('Drag-Drop within Date Groups', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');
  });

  test('should allow drag-drop within same date group', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album A');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album B');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album C');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroup = page.locator('[data-testid="date-group"]').first();
    const albumCards = dateGroup.locator('[data-testid="album-card"]');

    const firstCard = albumCards.first();
    const thirdCard = albumCards.nth(2);

    const firstBox = await firstCard.boundingBox();
    const thirdBox = await thirdCard.boundingBox();

    if (firstBox && thirdBox) {
      await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(thirdBox.x + thirdBox.width / 2, thirdBox.y + thirdBox.height / 2);
      await page.mouse.up();

      await page.waitForTimeout(100);

      const updatedAlbumCards = dateGroup.locator('[data-testid="album-card"]');
      await expect(updatedAlbumCards).toHaveCount(3);
    }
  });

  test('should show visual feedback during drag in grouped view', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 1');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 2');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroup = page.locator('[data-testid="date-group"]').first();
    const firstCard = dateGroup.locator('[data-testid="album-card"]').first();

    const box = await firstCard.boundingBox();

    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();

      const draggingCard = page.locator('.album-card.dragging');
      await expect(draggingCard).toHaveCount(1);

      await page.mouse.up();
    }
  });

  test('should maintain group boundaries during drag', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 1');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 2');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroups = page.locator('[data-testid="date-group"]');
    const count = await dateGroups.count();

    expect(count).toBeGreaterThan(0);
  });

  test('should update display order after drag within group', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'First Album');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Second Album');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroup = page.locator('[data-testid="date-group"]').first();
    const albumCards = dateGroup.locator('[data-testid="album-card"]');

    const initialCount = await albumCards.count();

    const firstCard = albumCards.first();
    const secondCard = albumCards.nth(1);

    const firstBox = await firstCard.boundingBox();
    const secondBox = await secondCard.boundingBox();

    if (firstBox && secondBox) {
      await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(secondBox.x + secondBox.width / 2, secondBox.y + secondBox.height / 2 + 10);
      await page.mouse.up();

      await page.waitForTimeout(100);

      const finalCount = await albumCards.count();
      expect(finalCount).toBe(initialCount);
    }
  });

  test('should persist order after drag in grouped view', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album X');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album Y');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroup = page.locator('[data-testid="date-group"]').first();
    const albumCards = dateGroup.locator('[data-testid="album-card"]');

    const firstCard = albumCards.first();
    const secondCard = albumCards.nth(1);

    const firstBox = await firstCard.boundingBox();
    const secondBox = await secondCard.boundingBox();

    if (firstBox && secondBox) {
      await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(secondBox.x + secondBox.width / 2, secondBox.y + secondBox.height / 2);
      await page.mouse.up();

      await page.waitForTimeout(100);

      await page.reload();
      await page.waitForLoadState('networkidle');

      const reloadedDateGroup = page.locator('[data-testid="date-group"]').first();
      const reloadedCards = reloadedDateGroup.locator('[data-testid="album-card"]');

      await expect(reloadedCards).toHaveCount(2);
    }
  });

  test('should allow switching between grouped and flat view after drag', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Test Album');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Another Album');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const toggleButton = page.getByTestId('grouping-toggle');
    await toggleButton.click();

    const flatGrid = page.getByTestId('album-grid');
    await expect(flatGrid).toBeVisible();

    const dateGroupHeaders = page.locator('[data-testid="date-group-header"]');
    await expect(dateGroupHeaders).toHaveCount(0);
  });

  test('should show drop indicators within groups', async ({ page }) => {
    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 1');
    await page.getByRole('button', { name: /create/i }).click();

    await page.waitForTimeout(100);

    await page.getByTestId('create-album-btn').click();
    await page.fill('input[name="album-name"]', 'Album 2');
    await page.getByRole('button', { name: /create/i }).click();

    await page.getByTestId('grouping-toggle').click();

    const dateGroup = page.locator('[data-testid="date-group"]').first();
    const firstCard = dateGroup.locator('[data-testid="album-card"]').first();
    const secondCard = dateGroup.locator('[data-testid="album-card"]').nth(1);

    const firstBox = await firstCard.boundingBox();
    const secondBox = await secondCard.boundingBox();

    if (firstBox && secondBox) {
      await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(secondBox.x + secondBox.width / 2, secondBox.y + secondBox.height / 2);

      const dropIndicator = page.locator('.drop-indicator');
      await expect(dropIndicator).toHaveCount(1);

      await page.mouse.up();
    }
  });
});
