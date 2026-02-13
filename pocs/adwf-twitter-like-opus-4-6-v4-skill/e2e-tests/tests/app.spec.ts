import { test, expect } from "@playwright/test";

test("login page loads", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();
  await expect(page.getByLabel("Username")).toBeVisible();
  await expect(page.getByLabel("Password")).toBeVisible();
  await expect(page.getByRole("button", { name: "Login" })).toBeVisible();
});

test("navigate to register page", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Register" }).click();
  await expect(page.getByRole("heading", { name: "Register" })).toBeVisible();
  await expect(page.getByLabel("Username")).toBeVisible();
  await expect(page.getByLabel("Email")).toBeVisible();
  await expect(page.getByLabel("Password")).toBeVisible();
});

test("register and login flow", async ({ page }) => {
  const uniqueUser = `e2euser_${Date.now()}`;
  await page.goto("/");
  await page.getByRole("button", { name: "Register" }).click();
  await page.getByLabel("Username").fill(uniqueUser);
  await page.getByLabel("Email").fill(`${uniqueUser}@test.com`);
  await page.getByLabel("Password").fill("password123");
  await page.getByRole("button", { name: "Register" }).click();
  await expect(page.getByText("Chirp")).toBeVisible({ timeout: 10000 });
});
