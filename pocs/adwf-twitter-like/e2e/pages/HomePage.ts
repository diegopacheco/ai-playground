import { Page, Locator } from '@playwright/test';

export class HomePage {
  readonly page: Page;
  readonly tweetTextarea: Locator;
  readonly tweetButton: Locator;
  readonly characterCount: Locator;
  readonly feedList: Locator;
  readonly navigationBar: Locator;
  readonly profileLink: Locator;
  readonly logoutButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.tweetTextarea = page.locator('textarea[aria-label="Tweet content"]');
    this.tweetButton = page.locator('button:has-text("Tweet")');
    this.characterCount = page.locator('text=/\\d+\\/280/');
    this.feedList = page.locator('.space-y-4').first();
    this.navigationBar = page.locator('nav');
    this.profileLink = page.locator('nav a:has-text("Profile")');
    this.logoutButton = page.locator('button:has-text("Logout")');
  }

  async goto() {
    await this.page.goto('/');
  }

  async createTweet(content: string) {
    await this.tweetTextarea.fill(content);
    await this.tweetButton.click();
    await this.page.waitForTimeout(1000);
    await this.page.reload();
    await this.page.waitForLoadState('networkidle');
  }

  async waitForTweetToAppear(content: string) {
    await this.page.waitForSelector(`text=${content}`);
  }

  async getTweetCards() {
    await this.page.waitForSelector('a[href^="/tweet/"]', { timeout: 15000 });
    await this.page.waitForTimeout(1000);
    return this.page.locator('.space-y-4 > div.border').all();
  }

  async likeTweet(index: number = 0) {
    const tweets = await this.getTweetCards();
    const likeButton = tweets[index].locator('button[aria-label*="Like"]');
    await likeButton.click();
    await this.page.waitForTimeout(1500);
  }

  async retweetTweet(index: number = 0) {
    const tweets = await this.getTweetCards();
    const retweetButton = tweets[index].locator('button[aria-label*="Retweet"]');
    await retweetButton.click();
    await this.page.waitForTimeout(1500);
  }

  async openTweetDetail(index: number = 0) {
    await this.page.waitForSelector('.space-y-4 > div.border', { timeout: 10000 });
    await this.page.waitForTimeout(500);
    const tweetCard = this.page.locator('.space-y-4 > div.border').nth(index);
    const paragraph = tweetCard.locator('p.mt-2').first();
    await paragraph.click();
    await this.page.waitForLoadState('networkidle');
  }

  async logout() {
    await this.logoutButton.click();
    await this.page.waitForURL('/login');
  }

  async goToProfile() {
    await this.profileLink.click();
  }
}
