import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';

test.describe('Responsive Design', () => {
  test('should display properly on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobileuser');

    await expect(page.locator('nav')).toBeVisible();
  });

  test('should display properly on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await createTestUser(page, 'tabletuser');

    await expect(page.locator('nav')).toBeVisible();
  });

  test('should display properly on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await createTestUser(page, 'desktopuser');

    await expect(page.locator('nav')).toBeVisible();
  });

  test('should have responsive tweet composer on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobilecomposeruser');
    const homePage = new HomePage(page);

    await expect(homePage.tweetTextarea).toBeVisible();
    await expect(homePage.tweetButton).toBeVisible();
  });

  test('should have responsive feed on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobilefeeduser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Mobile tweet ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    expect(tweets.length).toBeGreaterThan(0);
  });

  test('should have responsive navigation on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobilenavuser');

    const nav = page.locator('nav');
    await expect(nav).toBeVisible();

    const navContainer = nav.locator('.container');
    await expect(navContainer).toBeVisible();
  });

  test('should have scrollable feed on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobilescrolluser');
    const homePage = new HomePage(page);

    await homePage.createTweet('Tweet 1');
    await page.waitForTimeout(500);
    await homePage.createTweet('Tweet 2');
    await page.waitForTimeout(500);
    await homePage.createTweet('Tweet 3');
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    expect(tweets.length).toBeGreaterThanOrEqual(3);
  });

  test('should have responsive profile page on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobileprofileuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();

    await expect(page.locator('h1.text-2xl')).toBeVisible();
  });

  test('should have responsive tweet detail page on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobiledetailuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Mobile detail tweet ${Date.now()}`);
    await page.waitForTimeout(1000);

    await homePage.openTweetDetail(0);

    await expect(page.locator('textarea[aria-label="Comment content"]')).toBeVisible();
  });

  test('should have responsive login page on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    await page.goto('/login');

    await expect(page.locator('#email')).toBeVisible();
    await expect(page.locator('#password')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('should wrap long tweet text on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobilelongtextuser');
    const homePage = new HomePage(page);

    const longTweet = 'This is a very long tweet that should wrap properly on mobile devices to ensure good user experience and readability across different screen sizes';
    await homePage.createTweet(longTweet);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const tweetText = tweets[0].locator('p.mt-2.text-gray-900');

    await expect(tweetText).toBeVisible();
    const boundingBox = await tweetText.boundingBox();
    expect(boundingBox?.width).toBeLessThan(375);
  });

  test('should have touch-friendly buttons on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await createTestUser(page, 'mobilebuttonuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Touch test ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const likeButton = tweets[0].locator('button[aria-label*="Like"]');

    const boundingBox = await likeButton.boundingBox();
    expect(boundingBox?.height).toBeGreaterThan(20);
  });

  test('should maintain layout on orientation change', async ({ page }) => {
    await page.setViewportSize({ width: 667, height: 375 });
    await createTestUser(page, 'orientationuser');

    await expect(page.locator('nav')).toBeVisible();

    await page.setViewportSize({ width: 375, height: 667 });

    await expect(page.locator('nav')).toBeVisible();
  });
});
