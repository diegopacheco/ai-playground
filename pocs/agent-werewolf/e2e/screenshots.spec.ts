import { test } from "@playwright/test";
import path from "path";

const screenshotsDir = path.join(__dirname, "..", "screenshots");

test("capture setup page", async ({ page }) => {
  await page.goto("/");
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(screenshotsDir, "01-setup.png"), fullPage: true });
});

test("capture history page", async ({ page }) => {
  await page.goto("/history");
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(screenshotsDir, "02-history.png"), fullPage: true });
});
