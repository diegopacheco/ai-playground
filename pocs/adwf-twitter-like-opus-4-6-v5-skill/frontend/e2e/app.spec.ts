import { test, expect } from "@playwright/test";

test("login page renders", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("heading", { name: /chirper/i })).toBeVisible();
});

test("register page renders", async ({ page }) => {
  await page.goto("/register");
  await expect(page.getByRole("heading", { name: /join chirper/i })).toBeVisible();
});

test("navigation from login to register", async ({ page }) => {
  await page.goto("/login");
  await page.getByRole("link", { name: /register/i }).click();
  await expect(page).toHaveURL(/register/);
});

test("navigation from register to login", async ({ page }) => {
  await page.goto("/register");
  await page.getByRole("link", { name: /log in/i }).click();
  await expect(page).toHaveURL(/login/);
});
