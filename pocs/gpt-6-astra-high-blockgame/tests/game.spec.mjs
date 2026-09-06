import { test, expect } from "playwright/test";
async function ready(page) {
  await page.goto("/");
  await expect(page.locator("#game")).toHaveAttribute("data-ready", "true");
  await page.locator("#world").focus();
}
async function place(page) {
  const point = await page.evaluate(() => window.brisa.projectBlock(10, 0, 18));
  await page.mouse.click(point.x, point.y);
  return point;
}
test("A sunny world renders without browser errors and adapts to the viewport", async ({
  page,
}, info) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await ready(page);
  await expect(page.locator("#hotbar button")).toHaveCount(8);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page.evaluate(() => window.brisa.getState().blocks),
  ).toBeGreaterThan(10000);
  await page.screenshot({
    path: `printscreens/${info.project.name}.png`,
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
test("Players can choose blocks, build, remove, and reload a saved world", async ({
  page,
}) => {
  await ready(page);
  await page.getByRole("button", { name: "Terracotta", exact: true }).click();
  const before = await page.evaluate(() => window.brisa.getState());
  await place(page);
  await expect
    .poll(() => page.evaluate(() => window.brisa.getState().placed))
    .toBe(before.placed + 1);
  let state = await page.evaluate(() => window.brisa.getState());
  expect(state.edits.at(-1)[1]).toBe(2);
  await page.getByRole("button", { name: "Save world", exact: true }).click();
  await page.reload();
  await expect(page.locator("#game")).toHaveAttribute("data-ready", "true");
  expect((await page.evaluate(() => window.brisa.getState())).edits).toEqual(
    state.edits,
  );
  const coords = state.edits.at(-1)[0].split(",").map(Number);
  const point = await page.evaluate(
    (coords) => window.brisa.projectBlock(...coords),
    coords,
  );
  await page.mouse.click(point.x, point.y, { button: "right" });
  await expect
    .poll(() =>
      page.evaluate(() =>
        window.brisa.getState().edits.some((e) => e[1] === null),
      ),
    )
    .toBe(true);
});
test("Dragging rotates without placing blocks, and exploration moves the camera", async ({
  page,
}, info) => {
  await ready(page);
  const initial = await page.evaluate(() => window.brisa.getState());
  const rect = await page.locator("#world").boundingBox();
  await page.mouse.move(rect.x + rect.width * 0.5, rect.y + rect.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(
    rect.x + rect.width * 0.5 + 60,
    rect.y + rect.height * 0.4 + 30,
    { steps: 8 },
  );
  await page.mouse.up();
  expect((await page.evaluate(() => window.brisa.getState())).placed).toBe(
    initial.placed,
  );
  await page.locator("#explore-mode").click();
  const start = await page.evaluate(() => window.brisa.getState().camera);
  if (info.project.name === "mobile") {
    const button = page.getByRole("button", {
      name: "Move forward",
      exact: true,
    });
    const b = await button.boundingBox();
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
    await page.mouse.down();
    await expect
      .poll(() => page.evaluate(() => window.brisa.getState().explored))
      .toBe(true);
    await page.mouse.up();
  } else {
    await page.keyboard.down("w");
    await expect
      .poll(() => page.evaluate(() => window.brisa.getState().explored))
      .toBe(true);
    await page.keyboard.up("w");
  }
  expect(
    (await page.evaluate(() => window.brisa.getState())).camera,
  ).not.toEqual(start);
  await page.locator("#build-mode").click();
  expect((await page.evaluate(() => window.brisa.getState())).mode).toBe(
    "build",
  );
});
test("Help, sound, and resetting a world work", async ({ page }, info) => {
  await ready(page);
  await page.keyboard.press("?");
  await expect(page.locator("#help-dialog")).toBeVisible();
  await page.getByRole("button", { name: "Let’s make something" }).click();
  await page.getByRole("button", { name: "Enable ocean sound" }).click();
  await expect(page.locator("#sound")).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Mute ocean sound" }).click();
  await place(page);
  await expect
    .poll(() => page.evaluate(() => window.brisa.getState().placed))
    .toBe(1);
  await page.getByRole("button", { name: "Open block collection" }).click();
  await page
    .getByRole("button", { name: "Start a fresh world", exact: true })
    .click();
  await page.getByRole("button", { name: "Start fresh", exact: true }).click();
  expect((await page.evaluate(() => window.brisa.getState())).edits).toEqual(
    [],
  );
});
test("Malformed saved data cannot prevent a new world from loading", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem(
      "brisa-world-v1",
      '{"version":1,"edits":[["bad",400]]}',
    ),
  );
  await ready(page);
  expect((await page.evaluate(() => window.brisa.getState())).edits).toEqual(
    [],
  );
  await expect(page.locator("#toast")).toContainText("could not be read");
});

test("A world photo downloads as a PNG", async ({ page }, info) => {
  await ready(page);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    info.project.name === "mobile"
      ? page.getByRole("button", { name: "Take a photo" }).tap()
      : page.getByRole("button", { name: "Take a photo" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("brisa-costa-do-sol.png");
  expect(await download.failure()).toBeNull();
});
