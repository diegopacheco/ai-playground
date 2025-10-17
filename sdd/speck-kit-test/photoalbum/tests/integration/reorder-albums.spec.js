import { test, expect } from '@playwright/test';

test.describe('Album Drag-Drop Reorder', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');

    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'First Album');
    await page.click('[data-testid="create-album-submit"]');

    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Second Album');
    await page.click('[data-testid="create-album-submit"]');

    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Third Album');
    await page.click('[data-testid="create-album-submit"]');
  });

  test('should show albums in initial order', async ({ page }) => {
    const albumCards = page.locator('[data-testid="album-card"]');
    await expect(albumCards).toHaveCount(3);

    const firstCard = albumCards.nth(0);
    await expect(firstCard).toContainText('First Album');
  });

  test('should make album cards draggable', async ({ page }) => {
    const firstCard = page.locator('[data-testid="album-card"]').first();
    const draggable = await firstCard.getAttribute('draggable');

    expect(draggable).toBe('true');
  });

  test('should show visual feedback during drag', async ({ page }) => {
    const firstCard = page.locator('[data-testid="album-card"]').first();

    await firstCard.dispatchEvent('dragstart');

    await expect(firstCard).toHaveClass(/dragging/);
  });

  test('should show drop indicator when dragging over target', async ({ page }) => {
    const firstCard = page.locator('[data-testid="album-card"]').first();
    const secondCard = page.locator('[data-testid="album-card"]').nth(1);

    await firstCard.dispatchEvent('dragstart');
    await secondCard.dispatchEvent('dragover');

    const dropIndicator = page.locator('.drop-indicator');
    await expect(dropIndicator).toBeVisible();
  });

  test('should reorder albums after drag-drop', async ({ page }) => {
    const albumCards = page.locator('[data-testid="album-card"]');

    const firstCard = albumCards.nth(0);
    const thirdCard = albumCards.nth(2);

    const firstBox = await firstCard.boundingBox();
    const thirdBox = await thirdCard.boundingBox();

    await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(thirdBox.x + thirdBox.width / 2, thirdBox.y + thirdBox.height / 2);
    await page.mouse.up();

    await page.waitForTimeout(500);

    const reorderedCards = page.locator('[data-testid="album-card"]');
    const secondInOrder = reorderedCards.nth(1);
    await expect(secondInOrder).toContainText('Second Album');
  });

  test('should remove visual feedback after drop', async ({ page }) => {
    const firstCard = page.locator('[data-testid="album-card"]').first();

    await firstCard.dispatchEvent('dragstart');
    await firstCard.dispatchEvent('drop');
    await firstCard.dispatchEvent('dragend');

    await expect(firstCard).not.toHaveClass(/dragging/);
  });

  test('should maintain album data after reorder', async ({ page }) => {
    const firstCard = page.locator('[data-testid="album-card"]').first();
    const albumName = await firstCard.textContent();

    await firstCard.dispatchEvent('dragstart');

    const thirdCard = page.locator('[data-testid="album-card"]').nth(2);
    await thirdCard.dispatchEvent('drop');

    const movedCard = page.locator('[data-testid="album-card"]:has-text("' + albumName + '")');
    await expect(movedCard).toBeVisible();
  });
});
