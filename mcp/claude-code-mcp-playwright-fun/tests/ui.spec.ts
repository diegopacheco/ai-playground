import { test, expect } from '@playwright/test';

test.describe('Product Manager UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000');
  });

  test('page title is Product Manager', async ({ page }) => {
    await expect(page).toHaveTitle('Product Manager');
    await expect(page.locator('h1')).toHaveText('Product Manager');
  });

  test('Add New Product button shows and hides form', async ({ page }) => {
    const addButton = page.getByRole('button', { name: 'Add New Product' });
    await expect(addButton).toBeVisible();
    await addButton.click();
    await expect(page.getByRole('button', { name: 'Cancel' })).toBeVisible();
    await expect(page.getByPlaceholder('Product Name')).toBeVisible();
    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.getByRole('button', { name: 'Add New Product' })).toBeVisible();
    await expect(page.getByPlaceholder('Product Name')).not.toBeVisible();
  });

  test('product form has all fields', async ({ page }) => {
    await page.getByRole('button', { name: 'Add New Product' }).click();
    await expect(page.getByPlaceholder('Product Name')).toBeVisible();
    await expect(page.getByPlaceholder('Price')).toBeVisible();
    await expect(page.locator('input[type="date"]')).toBeVisible();
    await expect(page.getByPlaceholder('Product URL')).toBeVisible();
  });

  test('Save Product button submits new product', async ({ page }) => {
    const uniqueName = `Product-${Date.now()}`;
    const uniquePrice = Math.floor(Math.random() * 1000) + 100;
    await page.getByRole('button', { name: 'Add New Product' }).click();
    await page.getByPlaceholder('Product Name').fill(uniqueName);
    await page.getByPlaceholder('Price').fill(String(uniquePrice));
    await page.locator('input[type="date"]').fill('2024-06-01');
    await page.getByPlaceholder('Product URL').fill('https://test.com');
    await page.getByRole('button', { name: 'Save Product' }).click();
    await expect(page.getByRole('cell', { name: uniqueName })).toBeVisible();
    await expect(page.getByRole('cell', { name: `$${uniquePrice}` })).toBeVisible();
  });

  test('Cancel button hides form without saving', async ({ page }) => {
    await page.getByRole('button', { name: 'Add New Product' }).click();
    await page.getByPlaceholder('Product Name').fill('Should Not Save');
    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.getByPlaceholder('Product Name')).not.toBeVisible();
    await expect(page.getByRole('cell', { name: 'Should Not Save' })).not.toBeVisible();
  });

  test('products table has correct columns', async ({ page }) => {
    await expect(page.getByRole('columnheader', { name: 'Name' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Price' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Created At' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Action' })).toBeVisible();
  });

  test('pre-loaded products are displayed', async ({ page }) => {
    await expect(page.getByRole('cell', { name: 'iPhone 15' })).toBeVisible();
    await expect(page.getByRole('cell', { name: 'MacBook Pro' })).toBeVisible();
    await expect(page.getByRole('cell', { name: 'Apple Watch' })).toBeVisible();
  });

  test('View button exists on each row', async ({ page }) => {
    const rows = page.locator('tbody tr');
    const rowCount = await rows.count();
    const viewButtons = page.getByRole('button', { name: 'View' });
    await expect(viewButtons).toHaveCount(rowCount);
  });

  test('View button opens product URL in new tab', async ({ page, context }) => {
    const pagePromise = context.waitForEvent('page');
    await page.getByRole('row', { name: /iPhone 15/ }).getByRole('button', { name: 'View' }).click();
    const newPage = await pagePromise;
    await newPage.waitForLoadState();
    expect(newPage.url()).toContain('apple.com/iphone');
  });

  test('clicking table row opens product URL in new tab', async ({ page, context }) => {
    const pagePromise = context.waitForEvent('page');
    await page.getByRole('cell', { name: 'MacBook Pro' }).click();
    const newPage = await pagePromise;
    await newPage.waitForLoadState();
    expect(newPage.url()).toContain('apple.com/macbook-pro');
  });
});
