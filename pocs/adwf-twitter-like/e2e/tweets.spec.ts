import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';

test.describe('Tweets', () => {
  test('should create a new tweet', async ({ page }) => {
    await createTestUser(page, 'tweetuser');
    const homePage = new HomePage(page);

    const tweetContent = `Test tweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);

    await expect(page.locator(`text=${tweetContent}`)).toBeVisible({ timeout: 10000 });
  });

  test('should show character count when typing', async ({ page }) => {
    await createTestUser(page, 'charuser');
    const homePage = new HomePage(page);

    const content = 'Hello World';
    await homePage.tweetTextarea.fill(content);

    await expect(homePage.characterCount).toHaveText(`${content.length}/280`);
  });

  test('should disable tweet button when content is empty', async ({ page }) => {
    await createTestUser(page, 'emptyuser');
    const homePage = new HomePage(page);

    await expect(homePage.tweetButton).toBeDisabled();

    await homePage.tweetTextarea.fill('Some content');
    await expect(homePage.tweetButton).toBeEnabled();

    await homePage.tweetTextarea.fill('');
    await expect(homePage.tweetButton).toBeDisabled();
  });

  test('should enforce 280 character limit', async ({ page }) => {
    await createTestUser(page, 'limituser');
    const homePage = new HomePage(page);

    const longContent = 'a'.repeat(281);
    await homePage.tweetTextarea.fill(longContent);

    const actualContent = await homePage.tweetTextarea.inputValue();
    expect(actualContent.length).toBeLessThanOrEqual(280);
  });

  test('should like a tweet', async ({ page }) => {
    await createTestUser(page, 'likeuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet to like ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    await expect(tweetCard.locator('button[aria-label="Like"]')).toBeVisible();

    await tweetCard.locator('button[aria-label="Like"]').click();
    await page.waitForTimeout(1500);

    await expect(tweetCard.locator('button[aria-label="Unlike"]')).toBeVisible();
  });

  test('should unlike a tweet', async ({ page }) => {
    await createTestUser(page, 'unlikeuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet to unlike ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    await tweetCard.locator('button[aria-label="Like"]').click();
    await page.waitForTimeout(1500);

    await expect(tweetCard.locator('button[aria-label="Unlike"]')).toBeVisible();

    await tweetCard.locator('button[aria-label="Unlike"]').click();
    await page.waitForTimeout(1500);

    await expect(tweetCard.locator('button[aria-label="Like"]')).toBeVisible();
  });

  test('should retweet a tweet', async ({ page }) => {
    await createTestUser(page, 'retweetuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet to retweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    await expect(tweetCard.locator('button[aria-label="Retweet"]')).toBeVisible();

    await tweetCard.locator('button[aria-label="Retweet"]').click();
    await page.waitForTimeout(1500);

    await expect(tweetCard.locator('button[aria-label="Undo Retweet"]')).toBeVisible();
  });

  test('should navigate to tweet detail page', async ({ page }) => {
    await createTestUser(page, 'detailuser');
    const homePage = new HomePage(page);

    const tweetContent = `Navigable tweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetLink = page.locator('p.mt-2.text-gray-900', { hasText: tweetContent });
    await tweetLink.click();
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveURL(/\/tweet\/\d+/);
    await expect(page.locator(`text=${tweetContent}`)).toBeVisible();
  });

  test('should delete own tweet', async ({ page }) => {
    await createTestUser(page, 'deleteuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet to delete ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const deleteButton = tweetCard.locator('button[aria-label="Delete tweet"]');
    await deleteButton.click();
    await page.waitForTimeout(1000);

    await expect(page.locator(`text=${tweetContent}`)).not.toBeVisible();
  });

  test('should clear textarea after successful tweet', async ({ page }) => {
    await createTestUser(page, 'clearuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Test tweet ${Date.now()}`);
    await page.waitForTimeout(1000);

    const textareaValue = await homePage.tweetTextarea.inputValue();
    expect(textareaValue).toBe('');
  });

  test('should show loading state while posting tweet', async ({ page }) => {
    await createTestUser(page, 'loadinguser');
    const homePage = new HomePage(page);

    await page.route('**/api/tweets', async (route) => {
      if (route.request().method() === 'POST') {
        await new Promise(r => setTimeout(r, 1000));
        await route.continue();
      } else {
        await route.continue();
      }
    });

    await homePage.tweetTextarea.fill(`Loading test ${Date.now()}`);
    await homePage.tweetButton.click();

    await expect(homePage.tweetButton).toHaveText('Posting...');

    await page.unroute('**/api/tweets');
    await page.waitForTimeout(2000);
  });
});
