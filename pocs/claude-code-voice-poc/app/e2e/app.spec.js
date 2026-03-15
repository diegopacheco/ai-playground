import { test, expect } from '@playwright/test'

test.describe('Pokemon Battle Arena', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173')
  })

  test('should show home page with title', async ({ page }) => {
    await expect(page.locator('.title')).toContainText('Pokemon Battle Arena')
  })

  test('should have navigation tabs', async ({ page }) => {
    await expect(page.locator('.tab-nav')).toBeVisible()
    await expect(page.locator('.tab-link')).toHaveCount(5)
  })

  test('should show player name form', async ({ page }) => {
    await expect(page.locator('input').first()).toBeVisible()
    await expect(page.locator('.start-btn')).toBeVisible()
  })

  test('should navigate to cards after entering names', async ({ page }) => {
    await page.locator('input').first().fill('Ash')
    await page.locator('input').nth(1).fill('Gary')
    await page.locator('.start-btn').click()
    await expect(page).toHaveURL(/\/cards/)
  })

  test('should show pokedex page', async ({ page }) => {
    await page.locator('.tab-link', { hasText: 'Pokedex' }).click()
    await expect(page).toHaveURL(/\/pokedex/)
    await expect(page.locator('.pokedex-page h2')).toContainText('Pokedex')
  })

  test('should show battle history page', async ({ page }) => {
    await page.locator('.tab-link', { hasText: 'History' }).click()
    await expect(page).toHaveURL(/\/history/)
    await expect(page.locator('.no-history')).toContainText('No battles yet')
  })

  test('should filter pokemon by type in pokedex', async ({ page }) => {
    await page.locator('.tab-link', { hasText: 'Pokedex' }).click()
    await page.waitForSelector('.type-filter-btn')
    await page.locator('.type-filter-btn.fire').click()
    await expect(page.locator('.type-filter-btn.fire')).toHaveClass(/selected/)
  })
})
