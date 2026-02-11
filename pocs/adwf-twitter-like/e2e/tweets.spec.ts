import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';

test.describe('Tweets', () => {
  test('should create a new tweet', async ({ page }) => {
    await createTestUser(page, 'tweetuser');
    const homePage = new HomePage(page);

    const tweetContent = `Test tweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);

    await expect(page.locator(`text=${tweetContent}`)).toBeVisible();
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

    await homePage.createTweet(`Tweet to like ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const likeButton = tweets[0].locator('button[aria-label*="Like"]');
    const initialClass = await likeButton.getAttribute('class');

    await homePage.likeTweet(0);

    const updatedClass = await likeButton.getAttribute('class');
    expect(updatedClass).toContain('text-red-500');
    expect(initialClass).not.toContain('text-red-500');
  });

  test('should unlike a tweet', async ({ page }) => {
    await createTestUser(page, 'unlikeuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet to unlike ${Date.now()}`);
    await page.waitForTimeout(1000);

    await homePage.likeTweet(0);
    await page.waitForTimeout(500);

    const tweets = await homePage.getTweetCards();
    const likeButton = tweets[0].locator('button[aria-label*="Unlike"]');
    await likeButton.click();
    await page.waitForTimeout(500);

    const updatedButton = tweets[0].locator('button[aria-label*="Like"]');
    const updatedClass = await updatedButton.getAttribute('class');
    expect(updatedClass).not.toContain('text-red-500');
  });

  test('should retweet a tweet', async ({ page }) => {
    await createTestUser(page, 'retweetuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet to retweet ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const retweetButton = tweets[0].locator('button[aria-label*="Retweet"]');

    await homePage.retweetTweet(0);

    const updatedClass = await retweetButton.getAttribute('class');
    expect(updatedClass).toContain('text-green-500');
  });

  test('should navigate to tweet detail page', async ({ page }) => {
    await createTestUser(page, 'detailuser');
    const homePage = new HomePage(page);

    const tweetContent = `Navigable tweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await homePage.openTweetDetail(0);

    await expect(page).toHaveURL(/\/tweet\/\d+/);
    await expect(page.locator(`text=${tweetContent}`)).toBeVisible();
  });

  test('should delete own tweet', async ({ page }) => {
    await createTestUser(page, 'deleteuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet to delete ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const deleteButton = tweets[0].locator('button[aria-label="Delete tweet"]');
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

    await homePage.tweetTextarea.fill(`Loading test ${Date.now()}`);

    const tweetButtonPromise = homePage.tweetButton.click();

    await expect(homePage.tweetButton).toHaveText('Posting...');

    await tweetButtonPromise;
  });
});
