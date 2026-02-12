import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';
import { TweetDetailPage } from './pages/TweetDetailPage';

test.describe('Comments', () => {
  test('should add a comment to a tweet', async ({ page }) => {
    await createTestUser(page, 'commentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `Tweet with comments ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    const commentContent = `Test comment ${Date.now()}`;
    await detailPage.addComment(commentContent);

    await expect(page.locator(`text=${commentContent}`)).toBeVisible();
  });

  test('should show character count when typing comment', async ({ page }) => {
    await createTestUser(page, 'commentcharuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `CharCountTweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    const content = 'Test comment';
    await detailPage.commentTextarea.fill(content);

    await expect(detailPage.characterCount).toHaveText(`${content.length}/280`);
  });

  test('should disable comment button when content is empty', async ({ page }) => {
    await createTestUser(page, 'emptycommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `EmptyBtnTweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    await expect(detailPage.commentButton).toBeDisabled();

    await detailPage.commentTextarea.fill('Some content');
    await expect(detailPage.commentButton).toBeEnabled();

    await detailPage.commentTextarea.fill('');
    await expect(detailPage.commentButton).toBeDisabled();
  });

  test('should enforce 280 character limit for comments', async ({ page }) => {
    await createTestUser(page, 'limitcommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `LimitTweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    const longContent = 'a'.repeat(281);
    await detailPage.commentTextarea.fill(longContent);

    const actualContent = await detailPage.commentTextarea.inputValue();
    expect(actualContent.length).toBeLessThanOrEqual(280);
  });

  test('should show multiple comments on a tweet', async ({ page }) => {
    await createTestUser(page, 'multicommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `Tweet with multiple comments ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    await detailPage.addComment('First comment');
    await detailPage.addComment('Second comment');
    await detailPage.addComment('Third comment');

    const commentCount = await detailPage.getCommentCount();
    expect(commentCount).toBeGreaterThanOrEqual(3);
  });

  test('should delete own comment', async ({ page }) => {
    await createTestUser(page, 'deletecommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `Delete comment tweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    const commentContent = `Comment to delete ${Date.now()}`;
    await detailPage.addComment(commentContent);

    const initialCount = await detailPage.getCommentCount();

    const commentCard = page.locator('.border.border-gray-200.rounded-lg', { hasText: commentContent });
    await commentCard.locator('button:has-text("Delete")').click();
    await page.waitForTimeout(2000);
    await page.reload();
    await page.waitForLoadState('networkidle');

    const updatedCount = await detailPage.getCommentCount();
    expect(updatedCount).toBe(initialCount - 1);
  });

  test('should clear textarea after successful comment', async ({ page }) => {
    await createTestUser(page, 'clearcommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `ClearCommentTweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    await detailPage.addComment(`Comment ${Date.now()}`);

    const textareaValue = await detailPage.commentTextarea.inputValue();
    expect(textareaValue).toBe('');
  });

  test('should show loading state while posting comment', async ({ page }) => {
    await createTestUser(page, 'loadingcommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `LoadingTweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    await page.route('**/api/tweets/*/comments', async (route) => {
      if (route.request().method() === 'POST') {
        await new Promise(r => setTimeout(r, 2000));
        try { await route.continue(); } catch {}
      } else {
        await route.continue();
      }
    });

    await detailPage.commentTextarea.fill(`Comment ${Date.now()}`);
    await detailPage.commentButton.click();

    const submitButton = page.locator('button[type="submit"]');
    await expect(submitButton).toHaveText('Posting...');

    await page.unroute('**/api/tweets/*/comments');
    await page.waitForTimeout(3000);
  });

  test('should update comment count on tweet card', async ({ page }) => {
    await createTestUser(page, 'countuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    const tweetContent = `CountTweet ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    await detailPage.addComment('First comment');
    await page.waitForTimeout(500);

    const commentCount = await detailPage.getCommentCountFromTweet();
    expect(commentCount).toBeGreaterThan(0);
  });

  test('should display "No comments yet" when no comments', async ({ page }) => {
    await createTestUser(page, 'nocommentsuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet without comments ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');

    await expect(page.locator('text=No comments yet')).toBeVisible();
  });
});
