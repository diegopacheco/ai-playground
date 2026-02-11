import { test, expect } from '@playwright/test';
import { createTestUser } from './helpers/auth';
import { HomePage } from './pages/HomePage';

test.describe('Navigation', () => {
  test('should display navigation bar on all pages', async ({ page }) => {
    await createTestUser(page, 'navbaruser');
    const homePage = new HomePage(page);

    await expect(homePage.navigationBar).toBeVisible();

    await homePage.goToProfile();
    await expect(homePage.navigationBar).toBeVisible();
  });

  test('should navigate to home from navigation bar', async ({ page }) => {
    await createTestUser(page, 'homenavuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();
    await expect(page).toHaveURL(/\/profile\/\d+/);

    await page.locator('nav a:has-text("Home")').click();
    await expect(page).toHaveURL('/');
  });

  test('should navigate to profile from navigation bar', async ({ page }) => {
    await createTestUser(page, 'profilenavuser');
    const homePage = new HomePage(page);

    await expect(page).toHaveURL('/');

    await homePage.goToProfile();
    await expect(page).toHaveURL(/\/profile\/\d+/);
  });

  test('should display user handle in navigation bar', async ({ page }) => {
    const user = await createTestUser(page, 'handlenavuser');

    await expect(page.locator('nav').locator(`text=@${user.username}`)).toBeVisible();
  });

  test('should navigate to login after logout', async ({ page }) => {
    await createTestUser(page, 'logoutnavuser');
    const homePage = new HomePage(page);

    await homePage.logout();
    await expect(page).toHaveURL('/login');
  });

  test('should have sticky navigation bar', async ({ page }) => {
    await createTestUser(page, 'stickynavuser');

    const nav = page.locator('nav');
    const classes = await nav.getAttribute('class');

    expect(classes).toContain('sticky');
  });

  test('should navigate to tweet detail from feed', async ({ page }) => {
    await createTestUser(page, 'detailnavuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet for navigation ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const commentLink = tweetCard.locator('a[href*="/tweet/"]').last();
    await commentLink.click();

    await expect(page).toHaveURL(/\/tweet\/\d+/);
  });

  test('should navigate to user profile from tweet card', async ({ page }) => {
    await createTestUser(page, 'usernavuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet for user nav ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    const tweetCard = page.locator('div.border', { hasText: tweetContent });
    const userLink = tweetCard.locator('a[href^="/profile/"]').first();
    await userLink.click();

    await expect(page).toHaveURL(/\/profile\/\d+/);
  });

  test('should navigate back from tweet detail to feed', async ({ page }) => {
    await createTestUser(page, 'backnavuser');
    const homePage = new HomePage(page);

    const tweetContent = `Tweet for back nav ${Date.now()}`;
    await homePage.createTweet(tweetContent);
    await page.waitForTimeout(1000);

    await page.locator('p.mt-2.text-gray-900', { hasText: tweetContent }).click();
    await page.waitForLoadState('networkidle');
    await expect(page).toHaveURL(/\/tweet\/\d+/);

    await page.locator('nav a:has-text("Home")').click();
    await expect(page).toHaveURL('/');
  });

  test('should show Twitter Clone branding in navigation', async ({ page }) => {
    await createTestUser(page, 'brandinguser');

    await expect(page.locator('nav a:has-text("Twitter Clone")')).toBeVisible();
  });

  test('should navigate to home when clicking Twitter Clone logo', async ({ page }) => {
    await createTestUser(page, 'logonavuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();
    await expect(page).toHaveURL(/\/profile\/\d+/);

    await page.locator('nav a:has-text("Twitter Clone")').click();
    await expect(page).toHaveURL('/');
  });

  test('should handle browser back button', async ({ page }) => {
    await createTestUser(page, 'backbuttonuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();
    await expect(page).toHaveURL(/\/profile\/\d+/);

    await page.goBack();
    await expect(page).toHaveURL('/');
  });

  test('should handle browser forward button', async ({ page }) => {
    await createTestUser(page, 'forwardbuttonuser');
    const homePage = new HomePage(page);

    await homePage.goToProfile();
    await page.goBack();
    await page.goForward();

    await expect(page).toHaveURL(/\/profile\/\d+/);
  });
});
