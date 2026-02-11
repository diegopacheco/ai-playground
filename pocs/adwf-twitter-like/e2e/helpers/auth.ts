import { Page } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

export interface TestUser {
  username: string;
  email: string;
  password: string;
}

export async function createTestUser(page: Page, username: string): Promise<TestUser> {
  const timestamp = Date.now();
  const user: TestUser = {
    username: `${username}_${timestamp}`,
    email: `${username}_${timestamp}@test.com`,
    password: 'password123',
  };

  const loginPage = new LoginPage(page);
  await loginPage.register(user.username, user.email, user.password);
  await loginPage.waitForNavigation();

  return user;
}

export async function loginAsUser(page: Page, user: TestUser) {
  const loginPage = new LoginPage(page);
  await loginPage.login(user.email, user.password);
  await loginPage.waitForNavigation();
}
