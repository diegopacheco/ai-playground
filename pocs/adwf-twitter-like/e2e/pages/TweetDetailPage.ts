import { Page, Locator } from '@playwright/test';

export class TweetDetailPage {
  readonly page: Page;
  readonly tweetCard: Locator;
  readonly commentTextarea: Locator;
  readonly commentButton: Locator;
  readonly commentsList: Locator;
  readonly characterCount: Locator;

  constructor(page: Page) {
    this.page = page;
    this.tweetCard = page.locator('.border.border-gray-200.rounded-lg').first();
    this.commentTextarea = page.locator('textarea[aria-label="Comment content"]');
    this.commentButton = page.locator('button:has-text("Comment")');
    this.commentsList = page.locator('.space-y-3');
    this.characterCount = page.locator('text=/\\d+\\/280/');
  }

  async goto(tweetId: number) {
    await this.page.goto(`/tweet/${tweetId}`);
  }

  async addComment(content: string) {
    await this.commentTextarea.fill(content);
    await this.commentButton.click();
    await this.page.waitForTimeout(1500);
  }

  async getCommentCount() {
    await this.page.waitForTimeout(500);
    const comments = await this.commentsList.locator('.border.border-gray-200.rounded-lg').all();
    return comments.length;
  }

  async deleteComment(index: number = 0) {
    const comments = await this.commentsList.locator('.border.border-gray-200.rounded-lg').all();
    const deleteButton = comments[index].locator('button:has-text("Delete")');
    await deleteButton.click();
    await this.page.waitForTimeout(500);
  }

  async getTweetContent() {
    return this.tweetCard.locator('p.mt-2.text-gray-900').textContent();
  }

  async getLikeCount() {
    const likeButton = this.tweetCard.locator('button[aria-label*="Like"]');
    const count = await likeButton.locator('span').last().textContent();
    return parseInt(count || '0', 10);
  }

  async getRetweetCount() {
    const retweetButton = this.tweetCard.locator('button[aria-label*="Retweet"]');
    const count = await retweetButton.locator('span').last().textContent();
    return parseInt(count || '0', 10);
  }

  async getCommentCountFromTweet() {
    const commentLink = this.tweetCard.locator('a[href*="/tweet/"]').last();
    const count = await commentLink.locator('span').textContent();
    return parseInt(count || '0', 10);
  }
}
