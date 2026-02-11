import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';
import { TweetDetailPage } from './pages/TweetDetailPage';

test.describe('Comments', () => {
  test('should add a comment to a tweet', async ({ page }) => {
    await createTestUser(page, 'commentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet with comments ${Date.now()}`);
    await page.waitForTimeout(1000);

    await homePage.openTweetDetail(0);

    const commentContent = `Test comment ${Date.now()}`;
    await detailPage.addComment(commentContent);

    await expect(page.locator(`text=${commentContent}`)).toBeVisible();
  });

  test('should show character count when typing comment', async ({ page }) => {
    await createTestUser(page, 'commentcharuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    const content = 'Test comment';
    await detailPage.commentTextarea.fill(content);

    await expect(detailPage.characterCount).toHaveText(`${content.length}/280`);
  });

  test('should disable comment button when content is empty', async ({ page }) => {
    await createTestUser(page, 'emptycommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

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

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    const longContent = 'a'.repeat(281);
    await detailPage.commentTextarea.fill(longContent);

    const actualContent = await detailPage.commentTextarea.inputValue();
    expect(actualContent.length).toBeLessThanOrEqual(280);
  });

  test('should show multiple comments on a tweet', async ({ page }) => {
    await createTestUser(page, 'multicommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet with multiple comments ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    await detailPage.addComment('First comment');
    await detailPage.addComment('Second comment');
    await detailPage.addComment('Third comment');

    const commentCount = await detailPage.getCommentCount();
    expect(commentCount).toBe(3);
  });

  test('should delete own comment', async ({ page }) => {
    await createTestUser(page, 'deletecommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    const commentContent = `Comment to delete ${Date.now()}`;
    await detailPage.addComment(commentContent);

    const initialCount = await detailPage.getCommentCount();
    await detailPage.deleteComment(0);

    const updatedCount = await detailPage.getCommentCount();
    expect(updatedCount).toBe(initialCount - 1);
  });

  test('should clear textarea after successful comment', async ({ page }) => {
    await createTestUser(page, 'clearcommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    await detailPage.addComment(`Comment ${Date.now()}`);

    const textareaValue = await detailPage.commentTextarea.inputValue();
    expect(textareaValue).toBe('');
  });

  test('should show loading state while posting comment', async ({ page }) => {
    await createTestUser(page, 'loadingcommentuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    await detailPage.commentTextarea.fill(`Comment ${Date.now()}`);

    const commentButtonPromise = detailPage.commentButton.click();

    await expect(detailPage.commentButton).toHaveText('Posting...');

    await commentButtonPromise;
  });

  test('should update comment count on tweet card', async ({ page }) => {
    await createTestUser(page, 'countuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    await detailPage.addComment('First comment');
    await page.waitForTimeout(500);

    const commentCount = await detailPage.getCommentCountFromTweet();
    expect(commentCount).toBeGreaterThan(0);
  });

  test('should display "No comments yet" when no comments', async ({ page }) => {
    await createTestUser(page, 'nocommentsuser');
    const homePage = new HomePage(page);
    const detailPage = new TweetDetailPage(page);

    await homePage.createTweet(`Tweet without comments ${Date.now()}`);
    await page.waitForTimeout(1000);
    await homePage.openTweetDetail(0);

    await expect(page.locator('text=No comments yet')).toBeVisible();
  });
});
