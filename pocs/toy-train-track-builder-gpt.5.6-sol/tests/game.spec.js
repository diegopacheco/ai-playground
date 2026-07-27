import { test, expect } from '@playwright/test'

test('keeps the full desktop builder working', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Build your railway' })).toBeVisible()
  await expect(page.locator('.piece-card')).toHaveCount(12)
  await expect(page.locator('#pieceCount')).toHaveText('24 pieces')
  await page.getByRole('button', { name: 'Night' }).click()
  await expect(page.getByRole('button', { name: 'Night' })).toHaveClass(/active/)
  await page.getByRole('button', { name: 'Day' }).click()
  await page.locator('[data-piece="curve"]').click()
  await expect(page.locator('[data-piece="curve"]')).toHaveClass(/active/)
  await page.getByRole('button', { name: /Rotate/ }).click()
  await expect(page.locator('[data-piece="curve"]')).toHaveAttribute('data-angle', '180°')
  await page.getByRole('tab', { name: /Simulate/ }).click()
  await expect(page.getByRole('heading', { name: 'All aboard' })).toBeVisible()
  await page.getByRole('button', { name: /Start train/ }).click()
  await expect(page.getByRole('button', { name: /Stop train/ })).toBeVisible()
  expect(errors).toEqual([])
})

test('builds and runs with touch controls on phone screens', async ({ browser }) => {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    deviceScaleFactor: 2
  })
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })

  await page.goto('/')
  await expect(page.locator('#worldPanel')).toBeHidden()
  const tray = await page.locator('.piece-grid').evaluate(element => ({
    width: element.clientWidth,
    scrollWidth: element.scrollWidth,
    touchAction: getComputedStyle(element.querySelector('.piece-card')).touchAction
  }))
  expect(tray.scrollWidth).toBeGreaterThan(tray.width)
  expect(tray.touchAction).toBe('pan-x')
  await page.locator('.piece-grid').hover()
  await page.mouse.wheel(180, 0)
  await expect.poll(() => page.locator('.piece-grid').evaluate(element => element.scrollLeft)).toBeGreaterThan(0)

  const buildScene = await page.locator('#scene').boundingBox()
  const buildPanel = await page.locator('#buildPanel').boundingBox()
  expect(buildScene.y).toBeGreaterThanOrEqual(58)
  expect(buildScene.y + buildScene.height).toBeLessThanOrEqual(buildPanel.y)

  await page.locator('#worldToggle').tap()
  await expect(page.locator('#worldPanel')).toBeVisible()
  await expect(page.locator('#worldToggle')).toHaveAttribute('aria-expanded', 'true')
  await page.getByRole('button', { name: 'Rain' }).tap()
  await page.getByRole('button', { name: 'Snow' }).tap()
  await page.locator('#worldClose').tap()
  await expect(page.locator('#worldPanel')).toBeHidden()

  const house = page.locator('[data-piece="house"]')
  await house.scrollIntoViewIfNeeded()
  await house.tap()
  await expect(house).toHaveClass(/active/)
  const pieceCount = page.locator('#pieceCount')
  const before = await pieceCount.textContent()
  let placed = false
  for (const position of [{ x: 195, y: 330 }, { x: 130, y: 360 }, { x: 260, y: 360 }, { x: 195, y: 430 }]) {
    await page.locator('#scene').tap({ position })
    if (await pieceCount.textContent() !== before) {
      placed = true
      break
    }
  }
  expect(placed).toBe(true)
  await page.getByRole('button', { name: /Rotate/ }).tap()
  await page.getByRole('button', { name: /Undo/ }).tap()
  await page.getByRole('button', { name: /Undo/ }).tap()
  await expect(pieceCount).toHaveText(before)

  await page.getByRole('tab', { name: /Simulate/ }).tap()
  await expect(page.getByRole('heading', { name: 'All aboard' })).toBeVisible()
  const simulateScene = await page.locator('#scene').boundingBox()
  const simulatePanel = await page.locator('#simulatePanel').boundingBox()
  expect(simulateScene.y + simulateScene.height).toBeLessThanOrEqual(simulatePanel.y)
  await page.getByRole('button', { name: /Start train/ }).tap()
  await expect(page.getByRole('button', { name: /Stop train/ })).toBeVisible()

  await page.setViewportSize({ width: 844, height: 390 })
  const panel = await page.locator('#simulatePanel').boundingBox()
  expect(panel.height).toBeLessThan(300)
  expect(panel.y).toBeGreaterThan(50)
  expect(errors).toEqual([])
  await context.close()
})
