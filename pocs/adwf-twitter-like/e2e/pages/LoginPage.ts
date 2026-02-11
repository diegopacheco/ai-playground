import { Page, Locator } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly usernameInput: Locator;
  readonly submitButton: Locator;
  readonly toggleButton: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput = page.locator('#email');
    this.passwordInput = page.locator('#password');
    this.usernameInput = page.locator('#username');
    this.submitButton = page.locator('button[type="submit"]');
    this.toggleButton = page.locator('button.text-blue-500');
    this.errorMessage = page.locator('.bg-red-50');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async register(username: string, email: string, password: string) {
    await this.goto();
    const isLoginForm = await this.usernameInput.isVisible().catch(() => false);
    if (!isLoginForm) {
      await this.toggleButton.click();
      await this.usernameInput.waitFor({ state: 'visible' });
    }
    await this.usernameInput.fill(username);
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }

  async login(email: string, password: string) {
    await this.goto();
    const isLoginForm = await this.usernameInput.isVisible().catch(() => false);
    if (isLoginForm) {
      await this.toggleButton.click();
      await this.usernameInput.waitFor({ state: 'hidden' });
    }
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }

  async waitForNavigation() {
    await this.page.waitForURL('/');
  }

  async getErrorText() {
    return await this.errorMessage.textContent();
  }
}
