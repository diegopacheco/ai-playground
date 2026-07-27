import { test, expect } from '@playwright/test'

test('loads and unleashes the bode in Chrome', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'ANGRY BODE' })).toBeVisible()
  await expect(page.locator('#hero-goat')).toBeVisible()
  await page.getByRole('button', { name: /FRENETIC BANJO/ }).click()
  await expect(page.locator('#music-toggle')).toHaveAttribute('aria-pressed', 'false')
  await page.getByRole('button', { name: /FRENETIC BANJO/ }).click()
  await expect(page.locator('#music-toggle')).toHaveAttribute('aria-pressed', 'true')
  await page.screenshot({ path: '/tmp/angry-bode-start-v2.png' })
  await page.getByRole('button', { name: 'UNLEASH THE BODE!' }).click()
  await expect(page.locator('#start-screen')).toHaveClass(/leaving/)
  await page.keyboard.down('KeyD')
  await page.waitForTimeout(700)
  await page.keyboard.up('KeyD')
  await page.keyboard.down('KeyE')
  await page.waitForTimeout(500)
  await page.screenshot({ path: '/tmp/angry-bode-attack-v2.png' })
  await page.waitForTimeout(500)
  await page.keyboard.up('KeyE')
  await page.keyboard.down('ArrowUp')
  await page.waitForTimeout(4200)
  await page.keyboard.up('ArrowUp')
  await page.keyboard.press('KeyW')
  await page.waitForTimeout(900)
  await page.screenshot({ path: '/tmp/angry-bode-gameplay-v2.png' })
  await page.keyboard.press('KeyR')
  await expect(page.locator('#score')).toHaveText(/\d{6}/)
  await expect(page.locator('#score')).not.toHaveText('000000')
  expect(errors).toEqual([])
})

test('plays with touch controls on a mobile screen', async ({ browser }) => {
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
  await expect(page.locator('#touch-controls')).toBeVisible()
  await page.getByRole('button', { name: 'UNLEASH THE BODE!' }).tap()
  await expect(page.locator('#start-screen')).toHaveClass(/leaving/)
  const moveRight = page.getByRole('button', { name: 'Move right' })
  await moveRight.dispatchEvent('pointerdown', { pointerId: 1, pointerType: 'touch', isPrimary: true })
  await expect(moveRight).toHaveAttribute('data-active', 'true')
  await page.waitForTimeout(700)
  await moveRight.dispatchEvent('pointerup', { pointerId: 1, pointerType: 'touch', isPrimary: true })
  await expect(moveRight).toHaveAttribute('data-active', 'false')
  await page.getByRole('button', { name: 'J JUMP' }).tap()
  const laser = page.getByRole('button', { name: 'E LASER' })
  await laser.dispatchEvent('pointerdown', { pointerId: 2, pointerType: 'touch', isPrimary: true })
  await page.waitForTimeout(600)
  await laser.dispatchEvent('pointerup', { pointerId: 2, pointerType: 'touch', isPrimary: true })
  await page.getByRole('button', { name: 'W KICK' }).tap()
  await page.getByRole('button', { name: 'R BURP' }).tap()
  await page.getByRole('button', { name: 'Pause game' }).tap()
  await expect(page.locator('#pause-screen')).toBeVisible()
  await page.getByRole('button', { name: 'KEEP SMASHING' }).tap()
  await expect(page.locator('#pause-screen')).toBeHidden()
  expect(errors).toEqual([])
  await context.close()
})
