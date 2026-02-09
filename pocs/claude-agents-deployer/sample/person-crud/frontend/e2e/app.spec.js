const { test, expect } = require('@playwright/test');

test('page loads and shows Person CRUD heading', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Person CRUD' })).toBeVisible();
});

test('can navigate to Admin Panel', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Admin Panel' }).click();
  await expect(page.getByRole('heading', { name: /Admin Panel/ })).toBeVisible();
});

test('can navigate back to Person CRUD', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Admin Panel' }).click();
  await expect(page.getByRole('heading', { name: /Admin Panel/ })).toBeVisible();
  await page.getByRole('button', { name: 'Person CRUD' }).click();
  await expect(page.getByRole('heading', { name: 'Person CRUD' })).toBeVisible();
});

test('Admin Panel shows View Counts heading', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Admin Panel' }).click();
  await expect(page.getByRole('heading', { name: 'Admin Panel - View Counts' })).toBeVisible();
});

test('CRUD form has Name, Email, Age inputs and Add button', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByPlaceholder('Name')).toBeVisible();
  await expect(page.getByPlaceholder('Email')).toBeVisible();
  await expect(page.getByPlaceholder('Age')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Add' })).toBeVisible();
});
