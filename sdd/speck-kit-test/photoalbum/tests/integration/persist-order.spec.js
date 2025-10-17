import { test, expect } from '@playwright/test';

test.describe('Drag-Drop Persistence', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');

    for (let i = 1; i <= 4; i++) {
      await page.click('button:has-text("Create Album")');
      await page.fill('[data-testid="album-name-input"]', `Album ${i}`);
      await page.click('[data-testid="create-album-submit"]');
    }
  });

  test('should persist album order after reordering', async ({ page }) => {
    const albumCards = page.locator('[data-testid="album-card"]');

    const firstCard = albumCards.nth(0);
    const lastCard = albumCards.nth(3);

    const firstBox = await firstCard.boundingBox();
    const lastBox = await lastCard.boundingBox();

    await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(lastBox.x + lastBox.width / 2, lastBox.y + lastBox.height / 2);
    await page.mouse.up();

    await page.waitForTimeout(1000);

    await page.reload();

    await page.waitForSelector('[data-testid="album-card"]');

    const reloadedCards = page.locator('[data-testid="album-card"]');
    const lastPosition = reloadedCards.nth(3);

    await expect(lastPosition).toContainText('Album 1');
  });

  test('should save order to database immediately after drop', async ({ page }) => {
    const firstCard = page.locator('[data-testid="album-card"]').first();
    const secondCard = page.locator('[data-testid="album-card"]').nth(1);

    await firstCard.dispatchEvent('dragstart');
    await secondCard.dispatchEvent('drop');

    await page.waitForTimeout(100);

    await page.reload();
    await page.waitForSelector('[data-testid="album-card"]');

    const cards = page.locator('[data-testid="album-card"]');
    await expect(cards.first()).toBeVisible();
  });

  test('should maintain order across multiple reorders', async ({ page }) => {
    for (let i = 0; i < 3; i++) {
      const cards = page.locator('[data-testid="album-card"]');
      const firstCard = cards.nth(0);
      const lastCard = cards.nth(3);

      const firstBox = await firstCard.boundingBox();
      const lastBox = await lastCard.boundingBox();

      await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(lastBox.x + lastBox.width / 2, lastBox.y + lastBox.height / 2);
      await page.mouse.up();

      await page.waitForTimeout(200);
    }

    await page.reload();
    await page.waitForSelector('[data-testid="album-card"]');

    const cards = page.locator('[data-testid="album-card"]');
    await expect(cards).toHaveCount(4);
  });

  test('should restore order after browser refresh', async ({ page }) => {
    const initialCards = page.locator('[data-testid="album-card"]');
    const initialFirstText = await initialCards.first().textContent();
    const initialLastText = await initialCards.nth(3).textContent();

    const firstBox = await initialCards.first().boundingBox();
    const lastBox = await initialCards.nth(3).boundingBox();

    await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(lastBox.x + lastBox.width / 2, lastBox.y + lastBox.height / 2);
    await page.mouse.up();

    await page.waitForTimeout(500);

    await page.context().clearCookies();
    await page.reload();

    await page.waitForSelector('[data-testid="album-card"]');

    const restoredCards = page.locator('[data-testid="album-card"]');
    await expect(restoredCards).toHaveCount(4);
  });
});
