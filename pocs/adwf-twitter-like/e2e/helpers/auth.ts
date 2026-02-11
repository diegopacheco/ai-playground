import { Page } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

export interface TestUser {
  id: number;
  username: string;
  email: string;
  password: string;
}

export async function createTestUser(page: Page, username: string): Promise<TestUser> {
  const timestamp = Date.now();
  const user: TestUser = {
    id: 0,
    username: `${username}_${timestamp}`,
    email: `${username}_${timestamp}@test.com`,
    password: 'password123',
  };

  const loginPage = new LoginPage(page);
  await loginPage.register(user.username, user.email, user.password);
  await loginPage.waitForNavigation();

  const storedUser = await page.evaluate(() => {
    const data = localStorage.getItem('user');
    return data ? JSON.parse(data) : null;
  });
  if (storedUser) {
    user.id = storedUser.id;
  }

  return user;
}

export async function loginAsUser(page: Page, user: TestUser) {
  await page.evaluate(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  });
  const loginPage = new LoginPage(page);
  await loginPage.login(user.email, user.password);
  await loginPage.waitForNavigation();
}
