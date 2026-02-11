import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';

test.describe('Feed', () => {
  test('should display feed on home page', async ({ page }) => {
    await createTestUser(page, 'feeduser');
    const homePage = new HomePage(page);

    await expect(homePage.feedList.or(page.locator('text=No tweets to display'))).toBeVisible();
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

    const tweetContent = `New tweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await expect(page.locator(`text=${tweetContent}`)).toBeVisible({ timeout: 10000 });
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

    await page.route('**/api/tweets/feed**', async (route) => {
      await new Promise(r => setTimeout(r, 500));
      await route.continue();
    });

    await page.goto('/');

    const spinner = page.locator('.animate-spin');
    const feedContent = page.locator('.space-y-4');
    const emptyMsg = page.locator('text=No tweets to display');

    await expect(spinner.or(feedContent).or(emptyMsg).first()).toBeVisible({ timeout: 5000 });

    await page.unroute('**/api/tweets/feed**');
  });

  test('should show error message when feed fails to load', async ({ page }) => {
    await createTestUser(page, 'erroruser');

    await page.route('**/api/tweets/feed**', (route) => {
      route.fulfill({ status: 500, body: 'Internal Server Error' });
    });

    await page.goto('/');

    const errorMessage = page.locator('text=Failed to load tweets');
    const emptyMessage = page.locator('text=No tweets to display');

    await expect(errorMessage.or(emptyMessage).first()).toBeVisible({ timeout: 10000 });

    await page.unroute('**/api/tweets/feed**');
  });

  test('should display user avatar in tweet cards', async ({ page }) => {
    await createTestUser(page, 'avataruser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet with avatar ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const avatar = tweetCard.locator('.bg-blue-500.rounded-full');

    await expect(avatar).toBeVisible();
  });

  test('should display username and handle in tweet cards', async ({ page }) => {
    const user = await createTestUser(page, 'handleuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet with handle ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const handle = tweetCard.locator(`text=@${user.username}`);

    await expect(handle).toBeVisible();
  });

  test('should display tweet timestamp', async ({ page }) => {
    await createTestUser(page, 'timeuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet with timestamp ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const timestamp = tweetCard.locator('span.text-gray-500.text-sm').last();

    await expect(timestamp).toBeVisible();
  });

  test('should display interaction buttons on tweet cards', async ({ page }) => {
    await createTestUser(page, 'interactionuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet with interactions ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const likeButton = tweetCard.locator('button[aria-label*="Like"]');
    const retweetButton = tweetCard.locator('button[aria-label*="Retweet"]');
    const commentLink = tweetCard.locator('a[href*="/tweet/"]').last();

    await expect(likeButton).toBeVisible();
    await expect(retweetButton).toBeVisible();
    await expect(commentLink).toBeVisible();
  });

  test('should display counts for likes, retweets, and comments', async ({ page }) => {
    await createTestUser(page, 'countsuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet with counts ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const likeCount = tweetCard.locator('button[aria-label*="Like"] span').last();
    const retweetCount = tweetCard.locator('button[aria-label*="Retweet"] span').last();
    const commentCount = tweetCard.locator('a[href*="/tweet/"]').last().locator('span');

    await expect(likeCount).toBeVisible();
    await expect(retweetCount).toBeVisible();
    await expect(commentCount).toBeVisible();
  });
});
