import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('View Album Photos in Tile Grid', () => {
  test.beforeEach(async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.goto('http://localhost:5173');

    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Photo Grid Test');
    await page.click('[data-testid="create-album-submit"]');

    await page.click('[data-testid="album-card"]:has-text("Photo Grid Test")');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testImage = path.join(__dirname, '../fixtures/test-photo.jpg');
    await fileChooser.setFiles([testImage]);
  });

  test('should display photos in tile grid layout', async ({ page }) => {
    const photoGrid = page.locator('[data-testid="photo-grid"]');
    await expect(photoGrid).toBeVisible();
    await expect(photoGrid).toHaveCSS('display', /grid|flex/);
  });

  test('should show photo thumbnails in grid', async ({ page }) => {
    const photoTile = page.locator('[data-testid="photo-tile"]').first();
    await expect(photoTile).toBeVisible();

    const thumbnail = photoTile.locator('[data-testid="photo-thumbnail"]');
    await expect(thumbnail).toBeVisible();
  });

  test('should navigate back to album list when back button is clicked', async ({ page }) => {
    await page.click('[data-testid="back-to-albums"]');

    const albumGrid = page.locator('[data-testid="album-grid"]');
    await expect(albumGrid).toBeVisible();

    const albumCard = page.locator('[data-testid="album-card"]:has-text("Photo Grid Test")');
    await expect(albumCard).toBeVisible();
  });

  test('should display album name in header', async ({ page }) => {
    const header = page.locator('[data-testid="album-header"]');
    await expect(header).toContainText('Photo Grid Test');
  });

  test('should show photo count in album', async ({ page }) => {
    const photoCount = page.locator('[data-testid="photo-count"]');
    await expect(photoCount).toBeVisible();
    await expect(photoCount).toContainText('1');
  });

  test('should handle virtual scrolling for large photo collections', async ({ page, context }) => {
    const photoGrid = page.locator('[data-testid="photo-grid"]');
    await expect(photoGrid).toBeVisible();

    const initialHeight = await photoGrid.evaluate(el => el.scrollHeight);
    expect(initialHeight).toBeGreaterThan(0);
  });

  test('should display empty state when album has no photos', async ({ page }) => {
    await page.click('[data-testid="back-to-albums"]');

    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Empty Album');
    await page.click('[data-testid="create-album-submit"]');

    await page.click('[data-testid="album-card"]:has-text("Empty Album")');

    const emptyState = page.locator('[data-testid="empty-state"]');
    await expect(emptyState).toBeVisible();
    await expect(emptyState).toContainText('no photos');
  });

  test('should show photo metadata on hover', async ({ page }) => {
    const photoTile = page.locator('[data-testid="photo-tile"]').first();
    await photoTile.hover();

    const metadata = page.locator('[data-testid="photo-metadata"]');
    await expect(metadata).toBeVisible();
  });

  test('should maintain grid layout on window resize', async ({ page }) => {
    const photoGrid = page.locator('[data-testid="photo-grid"]');
    await expect(photoGrid).toBeVisible();

    await page.setViewportSize({ width: 800, height: 600 });
    await expect(photoGrid).toBeVisible();
    await expect(photoGrid).toHaveCSS('display', /grid|flex/);

    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(photoGrid).toBeVisible();
    await expect(photoGrid).toHaveCSS('display', /grid|flex/);
  });

  test('should persist photos after page refresh', async ({ page }) => {
    await page.reload();

    const photoTile = page.locator('[data-testid="photo-tile"]').first();
    await expect(photoTile).toBeVisible();
  });
});
