import { test, expect } from '@playwright/test';
import { createTestUser, loginAsUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';
import { ProfilePage } from './pages/ProfilePage';

test.describe('Follow/Unfollow', () => {
  test('should follow another user', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower1');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee1');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    const followButton = page.getByRole('button', { name: 'Follow', exact: true });
    await expect(followButton).toBeVisible();
    await followButton.click();
    await page.waitForTimeout(1000);
  });

  test('should show follower in followers list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower2');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee2');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    const followButton = page.getByRole('button', { name: 'Follow', exact: true });
    await followButton.click();
    await page.waitForTimeout(1000);

    await profilePage.switchToFollowersTab();
    await page.waitForTimeout(1000);

    await expect(page.locator(`.space-y-3 >> text=@${user1.username}`)).toBeVisible();
  });

  test('should show followed user in following list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower3');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee3');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    const followButton = page.getByRole('button', { name: 'Follow', exact: true });
    await followButton.click();
    await page.waitForTimeout(1000);

    await profilePage.goto(user1.id);
    await profilePage.switchToFollowingTab();
    await page.waitForTimeout(1000);

    await expect(page.locator(`text=@${user2.username}`)).toBeVisible();
  });

  test('should see followed user tweets in feed', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower4');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee4');
    const homePage2 = new HomePage(page2);

    const tweetContent = `Tweet from user2 ${Date.now()}`;
    await homePage2.createTweet(tweetContent);
    await page2.waitForTimeout(1000);
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    const followButton = page.getByRole('button', { name: 'Follow', exact: true });
    await followButton.click();
    await page.waitForTimeout(1000);

    await page.goto('/');
    await page.waitForTimeout(1000);

    await expect(page.locator(`text=${tweetContent}`)).toBeVisible();
  });

  test('should show empty feed when following user with no tweets', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'lonelyuser');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'emptyfollowee');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    await page.getByRole('button', { name: 'Follow', exact: true }).click();
    await page.waitForTimeout(1000);

    await page.goto('/');
    await page.waitForTimeout(1000);

    await expect(page.locator('text=No tweets to display')).toBeVisible();
  });

  test('should navigate to profile from followers list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower5');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee5');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    const followButton = page.getByRole('button', { name: 'Follow', exact: true });
    await followButton.click();
    await page.waitForTimeout(1000);

    await profilePage.switchToFollowersTab();
    await page.waitForTimeout(1000);

    const userLink = page.locator(`a[href*="/profile/"]`, { hasText: user1.username });
    await userLink.click();
    await page.waitForTimeout(500);

    await expect(page).toHaveURL(/\/profile\/\d+/);
  });

  test('should navigate to profile from following list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower6');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee6');
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    const profilePage = new ProfilePage(page);
    await profilePage.goto(user2.id);

    const followButton = page.getByRole('button', { name: 'Follow', exact: true });
    await followButton.click();
    await page.waitForTimeout(1000);

    await profilePage.goto(user1.id);
    await profilePage.switchToFollowingTab();
    await page.waitForTimeout(1000);

    const userLink = page.locator(`a[href*="/profile/"]`, { hasText: user2.username });
    await userLink.click();
    await page.waitForTimeout(500);

    await expect(page).toHaveURL(/\/profile\/\d+/);
  });
});
