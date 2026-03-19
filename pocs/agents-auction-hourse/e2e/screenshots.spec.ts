import { test, expect } from "@playwright/test";

test("01 - setup page", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector("text=Agent Auction House");
  await page.screenshot({
    path: "../screenshots/01-setup-page.png",
    fullPage: true,
  });
});

test("02 - setup page with agents selected", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector("text=Agent Auction House");

  const agents = page.locator('input[type="checkbox"]');
  await agents.nth(0).check();
  await agents.nth(1).check();
  await agents.nth(2).check();

  await page.waitForTimeout(500);
  await page.screenshot({
    path: "../screenshots/02-setup-agents-selected.png",
    fullPage: true,
  });
});

test("03 - history page empty", async ({ page }) => {
  await page.goto("/history");
  await page.waitForSelector("text=Auction History");
  await page.waitForTimeout(1000);
  await page.screenshot({
    path: "../screenshots/03-history-page.png",
    fullPage: true,
  });
});

test("04 - start auction and see live view", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector("text=Agent Auction House");

  const agents = page.locator('input[type="checkbox"]');
  await agents.nth(0).check();
  await agents.nth(1).check();
  await agents.nth(2).check();

  await page.click("text=Start Auction");
  await page.waitForTimeout(3000);
  await page.screenshot({
    path: "../screenshots/04-auction-live.png",
    fullPage: true,
  });
});

test("05 - wait for auction results", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector("text=Agent Auction House");

  const agents = page.locator('input[type="checkbox"]');
  await agents.nth(0).check();
  await agents.nth(1).check();
  await agents.nth(2).check();

  await page.click("text=Start Auction");
  await page.waitForTimeout(90000);
  await page.screenshot({
    path: "../screenshots/05-auction-results.png",
    fullPage: true,
  });
});
