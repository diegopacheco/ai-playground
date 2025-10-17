import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Add Photos to Album Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');

    await page.click('button:has-text("Create Album")');
    await page.fill('[data-testid="album-name-input"]', 'Test Album');
    await page.click('[data-testid="create-album-submit"]');

    await page.click('[data-testid="album-card"]:has-text("Test Album")');
  });

  test('should display add photos button in album view', async ({ page }) => {
    const addButton = page.locator('[data-testid="add-photos-button"]');
    await expect(addButton).toBeVisible();
  });

  test('should show file picker when add photos button is clicked', async ({ page }) => {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    expect(fileChooser).toBeDefined();
    expect(fileChooser.isMultiple()).toBe(true);
  });

  test('should display photos in tile grid after adding', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testImagePath = path.join(__dirname, '../fixtures/test-photo.jpg');
    await fileChooser.setFiles([testImagePath]);

    const photoGrid = page.locator('[data-testid="photo-grid"]');
    await expect(photoGrid).toBeVisible();

    const photoTile = page.locator('[data-testid="photo-tile"]').first();
    await expect(photoTile).toBeVisible();
  });

  test('should show photo thumbnails with correct dimensions', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testImagePath = path.join(__dirname, '../fixtures/test-photo.jpg');
    await fileChooser.setFiles([testImagePath]);

    const thumbnail = page.locator('[data-testid="photo-thumbnail"]').first();
    await expect(thumbnail).toBeVisible();

    const boundingBox = await thumbnail.boundingBox();
    expect(boundingBox.width).toBeLessThanOrEqual(600);
    expect(boundingBox.height).toBeLessThanOrEqual(600);
  });

  test('should add multiple photos at once', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testImage1 = path.join(__dirname, '../fixtures/test-photo.jpg');
    const testImage2 = path.join(__dirname, '../fixtures/test-photo-2.jpg');
    await fileChooser.setFiles([testImage1, testImage2]);

    const photoTiles = page.locator('[data-testid="photo-tile"]');
    await expect(photoTiles).toHaveCount(2);
  });

  test('should show loading indicator while processing photos', async ({ page }) => {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testImagePath = path.join(__dirname, '../fixtures/test-photo.jpg');
    const filesPromise = fileChooser.setFiles([testImagePath]);

    const loadingIndicator = page.locator('[data-testid="loading-indicator"]');
    await expect(loadingIndicator).toBeVisible();

    await filesPromise;
  });

  test('should show error message for unsupported file types', async ({ page }) => {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testFilePath = path.join(__dirname, '../fixtures/test-file.txt');
    await fileChooser.setFiles([testFilePath]);

    const errorMessage = page.locator('[data-testid="error-message"]');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText('unsupported');
  });

  test('should extract and display EXIF date from photos', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('[data-testid="add-photos-button"]');
    const fileChooser = await fileChooserPromise;

    const testImagePath = path.join(__dirname, '../fixtures/test-photo-with-exif.jpg');
    await fileChooser.setFiles([testImagePath]);

    const photoDate = page.locator('[data-testid="photo-date"]').first();
    await expect(photoDate).toBeVisible();
  });
});
