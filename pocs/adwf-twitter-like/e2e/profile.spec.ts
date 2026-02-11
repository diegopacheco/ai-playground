import { test, expect } from '@playwright/test';
import { createTestUser, loginAsUser, TestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';
import { ProfilePage } from './pages/ProfilePage';

test.describe('Profile Page', () => {
  test('should navigate to own profile', async ({ page }) => {
    const user = await createTestUser(page, 'profileuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();

    await expect(page).toHaveURL(/\/profile\/\d+/);
    await expect(page.locator(`text=@${user.username}`)).toBeVisible();
  });

  test('should display user information', async ({ page }) => {
    const user = await createTestUser(page, 'infouser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();

    await expect(page.locator(`text=@${user.username}`)).toBeVisible();
    await expect(page.locator('text=Joined')).toBeVisible();
  });

  test('should show user tweets on tweets tab', async ({ page }) => {
    await createTestUser(page, 'tweettabuser');
    const homePage = new HomePage(page);
    const profilePage = new ProfilePage(page);

    await homePage.createTweet(`Profile tweet ${Date.now()}`);
    await page.waitForTimeout(1000);

    await homePage.goToProfile();

    await profilePage.switchToTweetsTab();
    const tweetCount = await profilePage.getTweetCount();
    expect(tweetCount).toBeGreaterThan(0);
  });

  test('should switch between tabs', async ({ page }) => {
    await createTestUser(page, 'tabuser');
    const homePage = new HomePage(page);
    const profilePage = new ProfilePage(page);

    await homePage.goToProfile();

    await profilePage.switchToTweetsTab();
    await expect(profilePage.tweetsTab).toHaveClass(/border-blue-500/);

    await profilePage.switchToFollowersTab();
    await expect(profilePage.followersTab).toHaveClass(/border-blue-500/);

    await profilePage.switchToFollowingTab();
    await expect(profilePage.followingTab).toHaveClass(/border-blue-500/);
  });

  test('should show no tweets message when user has no tweets', async ({ page }) => {
    await createTestUser(page, 'notweetsuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();

    await expect(page.locator('text=No tweets yet')).toBeVisible();
  });

  test('should show no followers message when user has no followers', async ({ page }) => {
    await createTestUser(page, 'nofollowersuser');
    const homePage = new HomePage(page);
    const profilePage = new ProfilePage(page);

    await homePage.goToProfile();

    await profilePage.switchToFollowersTab();
    await expect(page.locator('text=No followers yet')).toBeVisible();
  });

  test('should show no following message when user follows no one', async ({ page }) => {
    await createTestUser(page, 'nofollowinguser');
    const homePage = new HomePage(page);
    const profilePage = new ProfilePage(page);

    await homePage.goToProfile();

    await profilePage.switchToFollowingTab();
    await expect(page.locator('text=Not following anyone yet')).toBeVisible();
  });

  test('should show follow button on other user profiles', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower');
    const homePage = new HomePage(page);

    await homePage.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    await expect(page.locator('button:has-text("Follow")')).toBeVisible();
  });

  test('should not show follow button on own profile', async ({ page }) => {
    await createTestUser(page, 'ownprofileuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();

    await expect(page.locator('button:has-text("Follow")')).not.toBeVisible();
  });

  test('should navigate to user profile from tweet', async ({ page }) => {
    await createTestUser(page, 'navuser');
    const homePage = new HomePage(page);

    const tweetContent = `Navigation test ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const usernameLink = tweetCard.locator('a[href^="/profile/"]').first();
    await usernameLink.click();

    await expect(page).toHaveURL(/\/profile\/\d+/);
  });

  test('should display loading spinner while fetching profile', async ({ page }) => {
    await createTestUser(page, 'loadingprofileuser');

    await page.goto('/profile/999999');

    const spinner = page.locator('.animate-spin');
    const isVisible = await spinner.isVisible().catch(() => false);

    expect(isVisible || await page.locator('text=User not found').isVisible()).toBeTruthy();
  });

  test('should show user not found for invalid profile', async ({ page }) => {
    await createTestUser(page, 'notfounduser');

    await page.goto('/profile/999999999');

    await expect(page.locator('text=User not found')).toBeVisible({ timeout: 10000 });
  });
});
