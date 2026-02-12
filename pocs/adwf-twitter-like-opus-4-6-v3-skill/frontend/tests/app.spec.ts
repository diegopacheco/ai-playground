import { test, expect } from "@playwright/test";

const unique = () => Math.random().toString(36).substring(2, 10);

test.describe("Register Flow", () => {
  test("should register a new user and redirect to home", async ({ page }) => {
    const suffix = unique();
    await page.goto("/register");

    await page.getByTestId("register-username").fill(`user${suffix}`);
    await page.getByTestId("register-email").fill(`user${suffix}@test.com`);
    await page.getByTestId("register-password").fill("password123");
    await page.getByTestId("register-submit").click();

    await expect(page).toHaveURL("/", { timeout: 10000 });
    await expect(page.getByRole("heading", { name: "Home" })).toBeVisible();
  });

  test("should show link to login page", async ({ page }) => {
    await page.goto("/register");
    const link = page.getByText("Sign in");
    await expect(link).toBeVisible();
    await link.click();
    await expect(page).toHaveURL("/login");
  });
});

test.describe("Login Flow", () => {
  test("should login and redirect to home", async ({ page, request }) => {
    const suffix = unique();
    await request.post("http://localhost:3000/api/auth/register", {
      data: {
        username: `login${suffix}`,
        email: `login${suffix}@test.com`,
        password: "testpass",
      },
    });

    await page.goto("/login");

    await page.getByTestId("login-email").fill(`login${suffix}@test.com`);
    await page.getByTestId("login-password").fill("testpass");
    await page.getByTestId("login-submit").click();

    await expect(page).toHaveURL("/", { timeout: 10000 });
    await expect(page.getByRole("heading", { name: "Home" })).toBeVisible();
  });

  test("should show link to register page", async ({ page }) => {
    await page.goto("/login");
    const link = page.getByText("Register");
    await expect(link).toBeVisible();
    await link.click();
    await expect(page).toHaveURL("/register");
  });
});

test.describe("Create Post", () => {
  test("should create a post and see it in timeline", async ({
    page,
    request,
  }) => {
    const suffix = unique();
    const res = await request.post("http://localhost:3000/api/auth/register", {
      data: {
        username: `poster${suffix}`,
        email: `poster${suffix}@test.com`,
        password: "testpass",
      },
    });
    const auth = await res.json();

    await page.goto("/login");
    await page.evaluate(
      ({ token, user }) => {
        localStorage.setItem("token", token);
        localStorage.setItem("user", JSON.stringify(user));
      },
      { token: auth.token, user: auth.user }
    );

    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Home" })).toBeVisible({ timeout: 10000 });

    const postText = `Test post ${suffix}`;
    await page.getByTestId("post-content").fill(postText);
    await page.getByTestId("post-submit").click();

    await expect(page.getByText(postText)).toBeVisible({ timeout: 10000 });
  });
});

test.describe("Like/Unlike", () => {
  test("should like and unlike a post", async ({ page, request }) => {
    const suffix = unique();
    const res = await request.post("http://localhost:3000/api/auth/register", {
      data: {
        username: `liker${suffix}`,
        email: `liker${suffix}@test.com`,
        password: "testpass",
      },
    });
    const auth = await res.json();

    await request.post("http://localhost:3000/api/posts", {
      headers: { Authorization: `Bearer ${auth.token}` },
      data: { content: `Likeable post ${suffix}` },
    });

    await page.goto("/login");
    await page.evaluate(
      ({ token, user }) => {
        localStorage.setItem("token", token);
        localStorage.setItem("user", JSON.stringify(user));
      },
      { token: auth.token, user: auth.user }
    );

    await page.goto("/");
    await expect(page.getByText(`Likeable post ${suffix}`)).toBeVisible({
      timeout: 10000,
    });

    const likeButton = page.getByTestId("like-button").first();
    await likeButton.click();
    await expect(likeButton).toContainText("1", { timeout: 5000 });

    await likeButton.click();
    await expect(likeButton).toContainText("0", { timeout: 5000 });
  });
});

test.describe("Navigation", () => {
  test("should navigate between pages", async ({ page, request }) => {
    const suffix = unique();
    const res = await request.post("http://localhost:3000/api/auth/register", {
      data: {
        username: `nav${suffix}`,
        email: `nav${suffix}@test.com`,
        password: "testpass",
      },
    });
    const auth = await res.json();

    await page.goto("/login");
    await page.evaluate(
      ({ token, user }) => {
        localStorage.setItem("token", token);
        localStorage.setItem("user", JSON.stringify(user));
      },
      { token: auth.token, user: auth.user }
    );

    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Home" })).toBeVisible({ timeout: 10000 });

    await page.getByText("Profile").click();
    await expect(page).toHaveURL(new RegExp("/profile/"), { timeout: 5000 });

    await page.getByTestId("nav-home").click();
    await expect(page).toHaveURL("/", { timeout: 5000 });

    await page.getByTestId("nav-logout").click();
    await expect(page).toHaveURL("/login", { timeout: 5000 });
  });

  test("should redirect to login when not authenticated", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page).toHaveURL("/login", { timeout: 5000 });
  });
});
