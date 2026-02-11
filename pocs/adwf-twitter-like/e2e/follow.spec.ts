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
    const user2Id = await page2.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    if (user2Id) {
      const profilePage = new ProfilePage(page);
      await profilePage.goto(user2Id);

      const followButton = page.locator('button:has-text("Follow")');
      await expect(followButton).toBeVisible();
      await followButton.click();
      await page.waitForTimeout(1000);
    }
  });

  test('should show follower in followers list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower2');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee2');
    const homePage2 = new HomePage(page2);
    await homePage2.goToProfile();

    const user2Id = await page2.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    if (user2Id) {
      const profilePage = new ProfilePage(page);
      await profilePage.goto(user2Id);

      const followButton = page.locator('button:has-text("Follow")');
      await followButton.click();
      await page.waitForTimeout(1000);

      await profilePage.switchToFollowersTab();
      await page.waitForTimeout(1000);

      await expect(page.locator(`text=@${user1.username}`)).toBeVisible();
    }
  });

  test('should show followed user in following list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower3');
    const homePage1 = new HomePage(page);

    const user1Id = await page.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee3');

    const user2Id = await page2.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    if (user2Id) {
      const profilePage = new ProfilePage(page);
      await profilePage.goto(user2Id);

      const followButton = page.locator('button:has-text("Follow")');
      await followButton.click();
      await page.waitForTimeout(1000);

      if (user1Id) {
        await profilePage.goto(user1Id);
        await profilePage.switchToFollowingTab();
        await page.waitForTimeout(1000);

        await expect(page.locator(`text=@${user2.username}`)).toBeVisible();
      }
    }
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

    await homePage2.goToProfile();
    const user2Id = await page2.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    if (user2Id) {
      const profilePage = new ProfilePage(page);
      await profilePage.goto(user2Id);

      const followButton = page.locator('button:has-text("Follow")');
      await followButton.click();
      await page.waitForTimeout(1000);

      await page.goto('/');
      await page.waitForTimeout(1000);

      await expect(page.locator(`text=${tweetContent}`)).toBeVisible();
    }
  });

  test('should show empty feed when not following anyone', async ({ page }) => {
    await createTestUser(page, 'lonelyuser');

    await expect(page.locator('text=No tweets to display')).toBeVisible();
  });

  test('should navigate to profile from followers list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower5');
    const homePage1 = new HomePage(page);

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee5');
    const homePage2 = new HomePage(page2);
    await homePage2.goToProfile();

    const user2Id = await page2.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    if (user2Id) {
      const profilePage = new ProfilePage(page);
      await profilePage.goto(user2Id);

      const followButton = page.locator('button:has-text("Follow")');
      await followButton.click();
      await page.waitForTimeout(1000);

      await profilePage.switchToFollowersTab();
      await page.waitForTimeout(1000);

      const userCard = page.locator(`text=@${user1.username}`).locator('..');
      await userCard.click();
      await page.waitForTimeout(500);

      await expect(page).toHaveURL(/\/profile\/\d+/);
    }
  });

  test('should navigate to profile from following list', async ({ page, context }) => {
    const user1 = await createTestUser(page, 'follower6');
    const homePage1 = new HomePage(page);

    const user1Id = await page.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });

    await homePage1.logout();

    const page2 = await context.newPage();
    const user2 = await createTestUser(page2, 'followee6');

    const user2Id = await page2.evaluate(() => {
      const url = window.location.href;
      const match = url.match(/profile\/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    });
    await page2.close();

    await loginAsUser(page, user1);
    await page.waitForTimeout(1000);

    if (user2Id && user1Id) {
      const profilePage = new ProfilePage(page);
      await profilePage.goto(user2Id);

      const followButton = page.locator('button:has-text("Follow")');
      await followButton.click();
      await page.waitForTimeout(1000);

      await profilePage.goto(user1Id);
      await profilePage.switchToFollowingTab();
      await page.waitForTimeout(1000);

      const userCard = page.locator(`text=@${user2.username}`).locator('..');
      await userCard.click();
      await page.waitForTimeout(500);

      await expect(page).toHaveURL(/\/profile\/\d+/);
    }
  });
});
