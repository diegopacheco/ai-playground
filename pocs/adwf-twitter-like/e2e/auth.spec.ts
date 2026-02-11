import { test, expect } from '@playwright/test';
import { LoginPage } from './pages/LoginPage';

test.describe('Authentication', () => {
  test('should register a new user successfully', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();

    await page.locator('text=Don\'t have an account? Sign up').click();
    await expect(loginPage.usernameInput).toBeVisible();

    const timestamp = Date.now();
    await loginPage.register(
      `testuser_${timestamp}`,
      `test_${timestamp}@example.com`,
      'password123'
    );

    await expect(page).toHaveURL('/');
    await expect(page.locator('nav')).toBeVisible();
  });

  test('should show error for duplicate email', async ({ page }) => {
    const loginPage = new LoginPage(page);
    const timestamp = Date.now();
    const email = `duplicate_${timestamp}@example.com`;

    await loginPage.register(`user1_${timestamp}`, email, 'password123');
    await expect(page).toHaveURL('/');

    await page.locator('button:has-text("Logout")').click();
    await expect(page).toHaveURL('/login');

    await loginPage.register(`user2_${timestamp}`, email, 'password123');
    await expect(loginPage.errorMessage).toBeVisible();
  });

  test('should login with valid credentials', async ({ page }) => {
    const loginPage = new LoginPage(page);
    const timestamp = Date.now();
    const email = `login_${timestamp}@example.com`;
    const username = `loginuser_${timestamp}`;

    await loginPage.register(username, email, 'password123');
    await expect(page).toHaveURL('/');

    await page.locator('button:has-text("Logout")').click();
    await expect(page).toHaveURL('/login');

    await loginPage.login(email, 'password123');
    await expect(page).toHaveURL('/');
    await expect(page.locator(`text=@${username}`)).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.login('invalid@example.com', 'wrongpassword');

    await expect(loginPage.errorMessage).toBeVisible();
  });

  test('should toggle between login and signup forms', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();

    await expect(loginPage.usernameInput).not.toBeVisible();

    await page.locator('text=Don\'t have an account? Sign up').click();
    await expect(loginPage.usernameInput).toBeVisible();

    await page.locator('text=Already have an account? Sign in').click();
    await expect(loginPage.usernameInput).not.toBeVisible();
  });

  test('should logout successfully', async ({ page }) => {
    const loginPage = new LoginPage(page);
    const timestamp = Date.now();

    await loginPage.register(
      `logoutuser_${timestamp}`,
      `logout_${timestamp}@example.com`,
      'password123'
    );

    await expect(page).toHaveURL('/');

    await page.locator('button:has-text("Logout")').click();
    await expect(page).toHaveURL('/login');
  });

  test('should redirect to login when not authenticated', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL('/login');
  });

  test('should validate password minimum length', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();

    await page.locator('text=Don\'t have an account? Sign up').click();

    const timestamp = Date.now();
    await loginPage.usernameInput.fill(`testuser_${timestamp}`);
    await loginPage.emailInput.fill(`test_${timestamp}@example.com`);
    await loginPage.passwordInput.fill('123');

    const validationMessage = await loginPage.passwordInput.evaluate(
      (el: HTMLInputElement) => el.validationMessage
    );

    expect(validationMessage).toBeTruthy();
  });
});
