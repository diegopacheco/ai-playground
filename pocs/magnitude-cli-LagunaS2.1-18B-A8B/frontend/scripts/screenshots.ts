import { chromium, FullPageScreenshotOptions } from "playwright";

const FRONTEND_URL = "http://localhost:5173";
const SCREENSHOT_DIR =
  "/Users/diegopacheco/git/diegopacheco/ai-playground/pocs/magnitude-cli-LagunaS2.1-18B-A8B/screenshots";

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  console.log("Navigating to frontend...");
  await page.goto(FRONTEND_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);

  console.log("Taking screenshot: home page (product table)...");
  await page.screenshot({
    path: `${SCREENSHOT_DIR}/home-page.png`,
    fullPage: true,
    type: "png",
  });

  console.log("Clicking 'Add Product' button...");
  await page.click("text=/\\+ Add Product/");
  await page.waitForTimeout(1000);

  console.log("Taking screenshot: product form modal...");
  await page.screenshot({
    path: `${SCREENSHOT_DIR}/product-form.png`,
    fullPage: true,
    type: "png",
  });

  console.log("Filling form and creating product...");
  await page.fill('input[id="name"]', "Gaming Monitor");
  await page.fill('textarea[id="description"]', "4K 32-inch gaming monitor with HDR");
  await page.fill('input[id="price"]', "599.99");
  await page.selectOption('select[id="category"]', "Electronics");
  await page.click('input[type="checkbox"][id="in_stock"]');

  console.log("Taking screenshot: filled form...");
  await page.screenshot({
    path: `${SCREENSHOT_DIR}/product-form-filled.png`,
    fullPage: true,
    type: "png",
  });

  console.log("Saving product...");
  await page.click("button:has-text('Create Product')");
  await page.waitForTimeout(2000);

  console.log("Taking screenshot: after creating product...");
  await page.screenshot({
    path: `${SCREENSHOT_DIR}/home-page-after-create.png`,
    fullPage: true,
    type: "png",
  });

  console.log("Editing first product...");
  const editButtons = await page.$$('button[title="Edit"]');
  if (editButtons.length > 0) {
    await editButtons[0].click();
    await page.waitForTimeout(1000);

    console.log("Taking screenshot: edit form...");
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/product-edit-form.png`,
      fullPage: true,
      type: "png",
    });

    await page.click('button:has-text("Update Product")');
    await page.waitForTimeout(1500);
  }

  console.log("Taking final screenshot...");
  await page.screenshot({
    path: `${SCREENSHOT_DIR}/final-view.png`,
    fullPage: true,
    type: "png",
  });

  await browser.close();
  console.log("All screenshots taken successfully!");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
