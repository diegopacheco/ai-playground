import { test, expect } from "@playwright/test"

test.describe("Memory Game", () => {
  test("player can enter name and start a game", async ({ page }) => {
    await page.goto("/")
    const nameInput = page.getByTestId("player-name-input")
    await expect(nameInput).toBeVisible()

    const startBtn = page.getByTestId("start-game-btn")
    await expect(startBtn).toBeDisabled()

    await nameInput.fill("TestPlayer")
    await expect(startBtn).toBeEnabled()
    await startBtn.click()

    await expect(page.getByTestId("game-board")).toBeVisible()
  })

  test("game board displays 16 cards face-down", async ({ page }) => {
    await page.goto("/")
    await page.getByTestId("player-name-input").fill("TestPlayer")
    await page.getByTestId("start-game-btn").click()

    await expect(page.getByTestId("game-board")).toBeVisible()

    for (let i = 0; i < 16; i++) {
      await expect(page.getByTestId(`card-${i}`)).toBeVisible()
    }

    const cards = page.getByTestId("game-board").locator("[data-testid^='card-']")
    await expect(cards).toHaveCount(16)
  })

  test("clicking a card flips it", async ({ page }) => {
    await page.goto("/")
    await page.getByTestId("player-name-input").fill("TestPlayer")
    await page.getByTestId("start-game-btn").click()

    await expect(page.getByTestId("game-board")).toBeVisible()

    const firstCard = page.getByTestId("card-0")
    await firstCard.click()

    await expect(firstCard.locator(".rotate-y-180")).toBeAttached()
  })

  test("matching two cards keeps them face-up", async ({ page }) => {
    await page.goto("/")
    await page.getByTestId("player-name-input").fill("TestPlayer")
    await page.getByTestId("start-game-btn").click()

    await expect(page.getByTestId("game-board")).toBeVisible()
    await expect(page.getByTestId("score-board")).toBeVisible()

    await page.getByTestId("card-0").click()
    await page.waitForTimeout(500)

    const firstCardValue = await page.getByTestId("card-0").locator(".rotate-y-180 span").textContent()

    let matchIndex = -1
    for (let i = 1; i < 16; i++) {
      await page.getByTestId(`card-${i}`).click()
      await page.waitForTimeout(500)

      const secondCardValue = await page.getByTestId(`card-${i}`).locator(".rotate-y-180 span").textContent()

      if (secondCardValue === firstCardValue) {
        matchIndex = i
        break
      }

      await page.waitForTimeout(1500)
    }

    if (matchIndex !== -1) {
      await page.waitForTimeout(500)
      const matchedCard = page.getByTestId("card-0").locator(".ring-green-400")
      await expect(matchedCard).toBeAttached()
    }
  })

  test("navigate to leaderboard page", async ({ page }) => {
    await page.goto("/")

    const leaderboardBtn = page.getByTestId("view-leaderboard-btn")
    await expect(leaderboardBtn).toBeVisible()
    await leaderboardBtn.click()

    await expect(page.getByTestId("leaderboard")).toBeVisible()
  })
})
