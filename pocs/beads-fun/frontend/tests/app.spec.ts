import { test, expect } from '@playwright/test'

const API = 'http://localhost:8080/api'
const uniqueUser = `user_${Date.now()}`
const password = 'testpass123'

test.describe.serial('Twitter Clone E2E', () => {
  test('shows login page by default', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.auth-title')).toHaveText('Twitter Clone')
    await expect(page.locator('.auth-box h2')).toHaveText('Login')
  })

  test('can switch to register form', async ({ page }) => {
    await page.goto('/')
    await page.click('.btn-switch')
    await expect(page.locator('.auth-box h2')).toHaveText('Register')
  })

  test('can register a new user', async ({ page }) => {
    await page.goto('/')
    await page.click('.btn-switch')
    await page.fill('input[placeholder="Username"]', uniqueUser)
    await page.fill('input[placeholder="Password"]', password)
    await page.click('.btn-tweet')
    await expect(page.locator('.header h1')).toHaveText('Home', { timeout: 5000 })
    await expect(page.locator('.header-user span')).toContainText(uniqueUser)
  })

  test('can logout', async ({ page }) => {
    await page.goto('/')
    await page.click('.btn-switch')
    await page.fill('input[placeholder="Username"]', `logout_${Date.now()}`)
    await page.fill('input[placeholder="Password"]', password)
    await page.click('.btn-tweet')
    await expect(page.locator('.header h1')).toHaveText('Home', { timeout: 5000 })
    await page.click('.btn-logout')
    await expect(page.locator('.auth-title')).toHaveText('Twitter Clone', { timeout: 5000 })
  })

  test('can login with existing user', async ({ page }) => {
    await page.goto('/')
    await page.fill('input[placeholder="Username"]', uniqueUser)
    await page.fill('input[placeholder="Password"]', password)
    await page.click('.btn-tweet')
    await expect(page.locator('.header h1')).toHaveText('Home', { timeout: 5000 })
  })

  test('shows error on wrong password', async ({ page }) => {
    await page.goto('/')
    await page.fill('input[placeholder="Username"]', uniqueUser)
    await page.fill('input[placeholder="Password"]', 'wrongpass')
    await page.click('.btn-tweet')
    await expect(page.locator('.auth-error')).toBeVisible({ timeout: 5000 })
  })

  test('can create a tweet', async ({ page }) => {
    await fetch(`${API}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: uniqueUser, password }),
    }).catch(() => {})
    const loginRes = await fetch(`${API}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: uniqueUser, password }),
    })
    const auth = await loginRes.json()

    await page.goto('/')
    await page.evaluate((a: { token: string; username: string }) => localStorage.setItem('auth', JSON.stringify(a)), auth)
    await page.reload()
    await expect(page.locator('.header h1')).toHaveText('Home', { timeout: 5000 })

    const tweetText = `Test tweet ${Date.now()}`
    await page.fill('textarea', tweetText)
    await page.locator('.compose-actions .btn-tweet').click()
    await expect(page.locator('.tweet-content').first()).toContainText(tweetText, { timeout: 10000 })
  })

  test('can like a tweet', async ({ page }) => {
    const loginRes = await fetch(`${API}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: uniqueUser, password }),
    })
    const auth = await loginRes.json()

    await page.goto('/')
    await page.evaluate((a: { token: string; username: string }) => localStorage.setItem('auth', JSON.stringify(a)), auth)
    await page.reload()
    await expect(page.locator('.tweet').first()).toBeVisible({ timeout: 10000 })
    await page.locator('.tweet-action').first().click()
    await expect(page.locator('.tweet-action.liked').first()).toBeVisible({ timeout: 5000 })
  })

  test('can search tweets', async ({ page }) => {
    const loginRes = await fetch(`${API}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: uniqueUser, password }),
    })
    const auth = await loginRes.json()

    await page.goto('/')
    await page.evaluate((a: { token: string; username: string }) => localStorage.setItem('auth', JSON.stringify(a)), auth)
    await page.reload()
    await expect(page.locator('.header h1')).toHaveText('Home', { timeout: 5000 })

    await page.fill('.search-bar input', 'nonexistent_xyz_query')
    await page.click('.btn-search')
    await expect(page.locator('.search-info')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('.empty-state')).toContainText('No tweets match')

    await page.click('.btn-clear')
    await expect(page.locator('.search-info')).not.toBeVisible()
  })

  test('can delete own tweet', async ({ page }) => {
    const loginRes = await fetch(`${API}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: uniqueUser, password }),
    })
    const auth = await loginRes.json()

    const tweetText = `delete_me_${Date.now()}`
    await fetch(`${API}/tweets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${auth.token}` },
      body: JSON.stringify({ content: tweetText }),
    })

    await page.goto('/')
    await page.evaluate((a: { token: string; username: string }) => localStorage.setItem('auth', JSON.stringify(a)), auth)
    await page.reload()
    await expect(page.locator('.tweet-content').first()).toContainText(tweetText, { timeout: 10000 })

    const tweetCount = await page.locator('.tweet').count()
    await page.locator('.tweet-action.delete').first().click()
    await expect(page.locator('.tweet')).toHaveCount(tweetCount - 1, { timeout: 10000 })
  })
})
