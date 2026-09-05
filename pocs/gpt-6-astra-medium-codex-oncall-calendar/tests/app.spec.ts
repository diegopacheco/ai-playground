import { test, expect } from "@playwright/test";
import { mkdir } from "node:fs/promises";

test("plans both years, filters roles, exports daily events and saves settings", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeEnabled();
  const year = new Date().getFullYear();
  await expect(page.locator(".month")).toHaveCount(12);
  await expect(page.locator(".day.primary").first()).toBeVisible();
  await mkdir("printscreens", { recursive: true });
  await page.screenshot({ path: "printscreens/calendar.png", fullPage: true });
  await page
    .getByRole("button", { name: String(year + 1), exact: true })
    .click();
  await expect(page.locator(".day.primary").first()).toBeVisible();
  await page.screenshot({ path: "printscreens/next-year.png", fullPage: true });
  await page.getByLabel("Filter role").selectOption("Secondary");
  await expect(page.locator(".day.primary")).toHaveCount(0);
  await expect(page.locator(".day.secondary").first()).toBeVisible();
  await page.getByLabel("Filter role").selectOption("All shifts");
  await page.getByRole("button", { name: "List view", exact: true }).click();
  await expect(page.locator(".week-row").first()).toBeVisible();
  await page.screenshot({ path: "printscreens/list.png", fullPage: true });
  await page.getByRole("button", { name: "Add to Google Calendar" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("dialog")).toContainText("4:00 – 6:00 AM");
  await page.screenshot({
    path: "printscreens/calendar-export.png",
    fullPage: true,
  });
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download calendar file" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("onward-oncall.ics");
  const calendar = await page.request.get(
    "/api/calendar.ics?" +
      new URLSearchParams({
        primary: await page.getByLabel("First primary date").inputValue(),
        secondary: await page.getByLabel("First secondary date").inputValue(),
        interval: "4",
        timezone: await page
          .getByLabel("Time zone", { exact: true })
          .inputValue(),
      }),
  );
  expect(calendar.ok()).toBeTruthy();
  const content = await calendar.text();
  expect(content).toContain("BEGIN:VEVENT");
  expect(content).toContain("SUMMARY:🔥 On Call Primary");
  expect(content).toContain("SUMMARY:🔥 On Call Secondary");
  await page.getByRole("button", { name: "Close calendar export" }).click();
  await page.getByLabel("Each role repeats every").fill("6");
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Generate schedule" }).click();
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeEnabled();
  await page.reload();
  await expect(page.getByLabel("Each role repeats every")).toHaveValue("6");
});

test("overlap errors prevent stale exports and recover after correction", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeEnabled();
  const secondary = await page.getByLabel("First secondary date").inputValue();
  await page
    .getByLabel("First secondary date")
    .fill(await page.getByLabel("First primary date").inputValue());
  await page.getByRole("button", { name: "Generate schedule" }).click();
  await expect(page.getByRole("alert")).toContainText("overlap");
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeDisabled();
  await page.getByLabel("First secondary date").fill(secondary);
  await page.getByRole("button", { name: "Generate schedule" }).click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeEnabled();
});

test("mobile layout keeps controls and calendar within the screen", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeEnabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({ path: "printscreens/mobile.png", fullPage: true });
});

test("API rejects malformed inputs with useful errors", async ({ request }) => {
  const missing = await request.get("/api/schedule");
  expect(missing.status()).toBe(400);
  expect(await missing.json()).toHaveProperty("error");
  const invalid = await request.get(
    "/api/schedule?primary=bad&secondary=2026-09-14&interval=4&timezone=UTC",
  );
  expect(invalid.status()).toBe(400);
  const missingPath = await request.get("/api/unknown");
  expect(missingPath.status()).toBe(404);
});

test("Google import sends separate timed events and skips previously imported events", async ({
  page,
}) => {
  await page.route("**/api/config", (route) =>
    route.fulfill({ json: { googleClientId: "test-client" } }),
  );
  await page.route("https://accounts.google.com/gsi/client", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: 'window.google = { accounts: { oauth2: { initTokenClient: options => ({ requestAccessToken: () => options.callback({ access_token: "test-token" }) }) } } };',
    }),
  );
  const events: {
    summary: string;
    id: string;
    start: { dateTime: string };
    end: { dateTime: string };
  }[] = [];
  await page.route(
    "https://www.googleapis.com/calendar/v3/calendars/primary/events",
    async (route) => {
      events.push(route.request().postDataJSON());
      await route.fulfill({
        status: events.length === 1 ? 409 : 200,
        json: {},
      });
    },
  );
  await page.goto("/");
  await page.getByLabel("Each role repeats every").fill("52");
  await page.getByRole("button", { name: "Generate schedule" }).click();
  await expect(
    page.getByRole("button", { name: "Add to Google Calendar" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Add to Google Calendar" }).click();
  await page
    .getByRole("button", { name: "Connect Google & add shifts" })
    .click();
  await expect(page.getByRole("status")).toContainText("All set.");
  await expect(page.getByRole("status")).toContainText("1 already present");
  expect(events.length).toBeGreaterThan(7);
  expect(new Set(events.map((event) => event.id)).size).toBe(events.length);
  expect(
    events.every(
      (event) =>
        event.start.dateTime.includes("T04:00:00") &&
        event.end.dateTime.includes("T06:00:00"),
    ),
  ).toBeTruthy();
  expect(
    events.some((event) => event.summary.startsWith("🔥 On Call Primary")),
  ).toBeTruthy();
  expect(
    events.some((event) => event.summary.startsWith("🔥 On Call Secondary")),
  ).toBeTruthy();
});
