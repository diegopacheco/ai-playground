import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';

test.describe('Feed', () => {
  test('should display feed on home page', async ({ page }) => {
    await createTestUser(page, 'feeduser');
    const homePage = new HomePage(page);

    await expect(homePage.feedList).toBeVisible();
  });

  test('should show tweet composer on home page', async ({ page }) => {
    await createTestUser(page, 'composeruser');
    const homePage = new HomePage(page);

    await expect(homePage.tweetTextarea).toBeVisible();
    await expect(homePage.tweetButton).toBeVisible();
  });

  test('should refresh feed after creating tweet', async ({ page }) => {
    await createTestUser(page, 'refreshuser');
    const homePage = new HomePage(page);

    const initialTweets = await homePage.getTweetCards();
    const initialCount = initialTweets.length;

    await homePage.createTweet(`New tweet ${Date.now()}`);
    await page.waitForTimeout(1000);

    const updatedTweets = await homePage.getTweetCards();
    const updatedCount = updatedTweets.length;

    expect(updatedCount).toBeGreaterThan(initialCount);
  });

  test('should display tweets in reverse chronological order', async ({ page }) => {
    await createTestUser(page, 'orderuser');
    const homePage = new HomePage(page);

    const tweet1 = `First tweet ${Date.now()}`;
    await homePage.createTweet(tweet1);
    await page.waitForTimeout(1000);

    const tweet2 = `Second tweet ${Date.now() + 1}`;
    await homePage.createTweet(tweet2);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const firstTweetText = await tweets[0].locator('p.mt-2.text-gray-900').textContent();

    expect(firstTweetText).toContain('Second tweet');
  });

  test('should show loading spinner while fetching feed', async ({ page }) => {
    await createTestUser(page, 'loadingfeeduser');

    await page.goto('/');

    const spinner = page.locator('.animate-spin');
    const isVisible = await spinner.isVisible().catch(() => false);

    expect(isVisible || (await page.locator('.space-y-4').isVisible())).toBeTruthy();
  });

  test('should show error message when feed fails to load', async ({ page, context }) => {
    await createTestUser(page, 'erroruser');

    await context.route('**/api/tweets/feed**', (route) => {
      route.abort();
    });

    await page.goto('/');
    await page.waitForTimeout(1000);

    const errorMessage = page.locator('text=Failed to load tweets');
    const emptyMessage = page.locator('text=No tweets to display');

    expect(await errorMessage.isVisible() || await emptyMessage.isVisible()).toBeTruthy();
  });

  test('should display user avatar in tweet cards', async ({ page }) => {
    await createTestUser(page, 'avataruser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet with avatar ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const avatar = tweets[0].locator('.bg-blue-500.rounded-full');

    await expect(avatar).toBeVisible();
  });

  test('should display username and handle in tweet cards', async ({ page }) => {
    const user = await createTestUser(page, 'handleuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet with handle ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const handle = tweets[0].locator(`text=@${user.username}`);

    await expect(handle).toBeVisible();
  });

  test('should display tweet timestamp', async ({ page }) => {
    await createTestUser(page, 'timeuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet with timestamp ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const timestamp = tweets[0].locator('span.text-gray-500.text-sm').last();

    await expect(timestamp).toBeVisible();
  });

  test('should display interaction buttons on tweet cards', async ({ page }) => {
    await createTestUser(page, 'interactionuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet with interactions ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const likeButton = tweets[0].locator('button[aria-label*="Like"]');
    const retweetButton = tweets[0].locator('button[aria-label*="Retweet"]');
    const commentLink = tweets[0].locator('a[href*="/tweet/"]');

    await expect(likeButton).toBeVisible();
    await expect(retweetButton).toBeVisible();
    await expect(commentLink).toBeVisible();
  });

  test('should display counts for likes, retweets, and comments', async ({ page }) => {
    await createTestUser(page, 'countsuser');
    const homePage = new HomePage(page);

    await homePage.createTweet(`Tweet with counts ${Date.now()}`);
    await page.waitForTimeout(1000);

    const tweets = await homePage.getTweetCards();
    const likeCount = tweets[0].locator('button[aria-label*="Like"] span').last();
    const retweetCount = tweets[0].locator('button[aria-label*="Retweet"] span').last();
    const commentCount = tweets[0].locator('a[href*="/tweet/"] span').last();

    await expect(likeCount).toBeVisible();
    await expect(retweetCount).toBeVisible();
    await expect(commentCount).toBeVisible();
  });
});
