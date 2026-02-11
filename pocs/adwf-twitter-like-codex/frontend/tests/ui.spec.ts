import { test, expect } from '@playwright/test'

const user = () => `u${Date.now()}${Math.floor(Math.random() * 100000)}`

async function createUser(page, username) {
  await page.getByPlaceholder('username').fill(username)
  const userCall = page.waitForResponse(r => r.url().includes('/api/users') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Create' }).click()
  await userCall
  await expect(page.getByPlaceholder('username')).toHaveValue('')
}

async function createPost(page, content) {
  await page.getByPlaceholder('what is happening').fill(content)
  const postCall = page.waitForResponse(r => r.url().includes('/api/posts') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Post' }).click()
  await postCall
}

test('page title', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173')
  await expect(page).toHaveTitle(/Twitter Like Clone/)
})

test('create user updates active user id', async ({ page }) => {
  const username = user()
  await page.goto('http://127.0.0.1:4173')
  await createUser(page, username)
  await expect(page.getByText(/Active user id: \d+/)).toBeVisible()
})

test('create post renders in timeline', async ({ page }) => {
  const username = user()
  await page.goto('http://127.0.0.1:4173')
  await createUser(page, username)
  await createPost(page, 'hello timeline')
  await expect(page.getByText('hello timeline')).toBeVisible()
  await expect(page.getByText(username)).toBeVisible()
})

test('like increments post counter', async ({ page }) => {
  const username = user()
  await page.goto('http://127.0.0.1:4173')
  await createUser(page, username)
  await createPost(page, 'like me')
  const row = page.locator('li', { hasText: 'like me' })
  await expect(row.getByText('likes: 0')).toBeVisible()
  await row.getByRole('button', { name: 'Like' }).click()
  await expect(row.getByText('likes: 1')).toBeVisible()
})

test('duplicate like shows error', async ({ page }) => {
  const username = user()
  await page.goto('http://127.0.0.1:4173')
  await createUser(page, username)
  await createPost(page, 'single like')
  const row = page.locator('li', { hasText: 'single like' })
  await row.getByRole('button', { name: 'Like' }).click()
  await row.getByRole('button', { name: 'Like' }).click()
  await expect(page.getByText('failed to like post')).toBeVisible()
})

test('duplicate username shows error', async ({ page }) => {
  const username = user()
  await page.goto('http://127.0.0.1:4173')
  await createUser(page, username)
  await page.getByPlaceholder('username').fill(username)
  await page.getByRole('button', { name: 'Create' }).click()
  await expect(page.getByText('failed to create user')).toBeVisible()
})
