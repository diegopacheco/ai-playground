import { Page, Locator } from '@playwright/test';

export class ProfilePage {
  readonly page: Page;
  readonly username: Locator;
  readonly displayName: Locator;
  readonly bio: Locator;
  readonly followButton: Locator;
  readonly tweetsTab: Locator;
  readonly followersTab: Locator;
  readonly followingTab: Locator;
  readonly tweetsList: Locator;
  readonly userCards: Locator;

  constructor(page: Page) {
    this.page = page;
    this.username = page.locator('text=/^@[a-zA-Z0-9_]+$/');
    this.displayName = page.locator('h1.text-2xl');
    this.bio = page.locator('p.mt-2.text-gray-700');
    this.followButton = page.getByRole('button', { name: 'Follow', exact: true });
    this.tweetsTab = page.locator('button:has-text("Tweets")');
    this.followersTab = page.locator('button:has-text("Followers")');
    this.followingTab = page.locator('button:has-text("Following")');
    this.tweetsList = page.locator('.space-y-4');
    this.userCards = page.locator('.space-y-3');
  }

  async goto(userId: number) {
    await this.page.goto(`/profile/${userId}`);
  }

  async followUser() {
    await this.followButton.click();
    await this.page.waitForTimeout(500);
  }

  async switchToTweetsTab() {
    await this.tweetsTab.click();
  }

  async switchToFollowersTab() {
    await this.followersTab.click();
    await this.page.waitForTimeout(500);
  }

  async switchToFollowingTab() {
    await this.followingTab.click();
    await this.page.waitForTimeout(500);
  }

  async getTweetCount() {
    return (await this.tweetsList.locator('.border.border-gray-200.rounded-lg').all()).length;
  }

  async getUserCardCount() {
    await this.page.waitForTimeout(500);
    const cards = await this.page.locator('.bg-white.rounded-lg.p-4').all();
    return cards.length;
  }
}
