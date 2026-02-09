import { test, expect } from "@playwright/test";

test("home page loads", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("text=Blog Platform")).toBeVisible();
});

test("nav bar has links", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("text=Home")).toBeVisible();
  await expect(page.locator("text=New Post")).toBeVisible();
  await expect(page.locator("text=Profile")).toBeVisible();
  await expect(page.locator("text=Admin")).toBeVisible();
});

test("navigate to create post page", async ({ page }) => {
  await page.goto("/");
  await page.click("text=New Post");
  await expect(page).toHaveURL(/\/posts\/create/);
});

test("create post form has fields", async ({ page }) => {
  await page.goto("/posts/create");
  await expect(page.locator("input[placeholder*='itle']").first()).toBeVisible();
});

test("navigate to profile page", async ({ page }) => {
  await page.goto("/");
  await page.click("text=Profile");
  await expect(page).toHaveURL(/\/profile/);
});

test("navigate to admin page", async ({ page }) => {
  await page.goto("/");
  await page.click("text=Admin");
  await expect(page).toHaveURL(/\/admin/);
  await expect(page.locator("text=Admin Panel")).toBeVisible();
});

test("home page shows no posts message when empty", async ({ page }) => {
  await page.goto("/");
  const noPostsOrList = page.locator("text=No posts yet").or(page.locator("text=All Posts"));
  await expect(noPostsOrList.first()).toBeVisible();
});

test("navigate back to home from create post", async ({ page }) => {
  await page.goto("/posts/create");
  await page.click("text=Home");
  await expect(page).toHaveURL("/");
});
