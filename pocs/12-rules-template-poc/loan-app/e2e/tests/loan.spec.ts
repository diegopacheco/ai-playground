import { test, expect } from '@playwright/test'

test('approved loan flow shows approval and computed terms', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Auto Loan' })).toBeVisible()

  await page.getByLabel('Loan amount ($)').fill('25000')
  await page.getByLabel('Term (months)').fill('60')
  await page.getByLabel('Annual income ($)').fill('80000')
  await page.getByLabel('Vehicle value ($)').fill('30000')
  await page.getByLabel('Credit score').fill('720')

  await page.getByRole('button', { name: /request loan/i }).click()

  await expect(page.locator('.result.approved')).toBeVisible()
  await expect(page.locator('.result .status')).toHaveText('Approved')
  await expect(page.getByText('$495.03')).toBeVisible()
  await expect(page.getByText('7.00%')).toBeVisible()
})

test('denied loan when credit score is below threshold', async ({ page }) => {
  await page.goto('/')

  await page.getByLabel('Loan amount ($)').fill('15000')
  await page.getByLabel('Term (months)').fill('60')
  await page.getByLabel('Annual income ($)').fill('80000')
  await page.getByLabel('Vehicle value ($)').fill('30000')
  await page.getByLabel('Credit score').fill('500')

  await page.getByRole('button', { name: /request loan/i }).click()

  await expect(page.locator('.result.denied')).toBeVisible()
  await expect(page.locator('.result .status')).toHaveText('Denied')
  await expect(page.getByText(/Credit score below minimum/)).toBeVisible()
})

test('denied loan when loan-to-value ratio is too high', async ({ page }) => {
  await page.goto('/')

  await page.getByLabel('Loan amount ($)').fill('28000')
  await page.getByLabel('Term (months)').fill('60')
  await page.getByLabel('Annual income ($)').fill('80000')
  await page.getByLabel('Vehicle value ($)').fill('30000')
  await page.getByLabel('Credit score').fill('720')

  await page.getByRole('button', { name: /request loan/i }).click()

  await expect(page.locator('.result.denied')).toBeVisible()
  await expect(page.getByText(/85% of vehicle value/)).toBeVisible()
})
