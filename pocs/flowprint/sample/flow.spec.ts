import { test, expect } from '@playwright/test';
import { pathToFileURL } from 'node:url';

test('Record browser flows and turn them into Playwright tests by Diego Pacheco', async ({ page }) => {
  await page.goto(pathToFileURL(`${process.cwd()}/calculator.html`).href);
  const search = page.getByRole('searchbox', { name: 'Search', exact: true });
  await search.fill('calculator');
  await search.press('Enter');
  await expect(page).toHaveURL(url => url.hash === '#calculator');
  await page.getByRole('button', { name: '5', exact: true }).click();
  await page.getByRole('button', { name: 'multiply', exact: true }).click();
  await page.getByRole('button', { name: '5', exact: true }).click();
  await page.getByRole('button', { name: 'equals', exact: true }).click();
  await expect(page.getByRole('status', { name: 'Result', exact: true })).toHaveText('25');
});
