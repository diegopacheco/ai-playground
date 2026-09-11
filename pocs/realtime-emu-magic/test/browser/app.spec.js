import { test, expect } from '@playwright/test'
import { mkdir } from 'node:fs/promises'
import { colorRom } from '../color-rom.mjs'

async function screenColor(page, canvas) {
  const box = await canvas.boundingBox()
  const pixel = await page.screenshot({ clip: { x: Math.floor(box.x + box.width / 2), y: Math.floor(box.y + box.height / 2), width: 1, height: 1 } })
  return page.evaluate(async encoded => {
    const image = new Image()
    image.src = `data:image/png;base64,${encoded}`
    await image.decode()
    const surface = document.createElement('canvas')
    surface.width = surface.height = 1
    const context = surface.getContext('2d')
    context.drawImage(image, 0, 0)
    return Array.from(context.getImageData(0, 0, 1, 1).data)
  }, pixel.toString('base64'))
}

async function startGame(page) {
  page.on('pageerror', error => process.stderr.write(`Player error: ${error.message}\n`))
  page.on('requestfailed', request => process.stderr.write(`Failed asset: ${request.url()} ${request.failure()?.errorText}\n`))
  page.on('console', entry => { if (entry.type() === 'error') process.stderr.write(`Browser error: ${entry.text()}\n`) })
  await page.goto('/')
  await page.locator('#romInput').setInputFiles({ name: 'fantasy-game.sfc', mimeType: 'application/octet-stream', buffer: colorRom() })
  const player = page.frameLocator('#player')
  await player.getByText('Start Game', { exact: true }).click()
  await expect(page.locator('#gameStatus')).toHaveText('NOW PLAYING')
  await expect(player.locator('canvas').first()).toBeVisible()
  return player
}

test('desktop and mobile cartridge screen', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  expect(await page.locator('link[rel="icon"]').evaluate(async element => {
    const icon = new Image()
    icon.src = element.href
    await icon.decode()
    return icon.naturalWidth > 0 && element.href.startsWith('data:image/svg+xml,')
  })).toBe(true)
  await expect(page.getByRole('button', { name: 'Ask Codex' })).toBeDisabled()
  await page.locator('#romInput').setInputFiles({ name: 'invalid.txt', mimeType: 'text/plain', buffer: Buffer.from('invalid') })
  await expect(page.getByRole('alert')).toContainText('Choose a non-empty')
  await page.reload()
  await mkdir('printscreens', { recursive: true })
  await page.screenshot({ path: 'printscreens/desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(page.getByRole('heading', { name: 'Play outside the rules.' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: 'printscreens/mobile.png', fullPage: true })
  expect(errors).toEqual([])
})

test('the console fits the viewport while only the conversation scrolls', async ({ page }) => {
  await page.goto('/')
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 390, height: 844 }, { width: 844, height: 390 }]) {
    await page.setViewportSize(viewport)
    for (const selector of ['#screen', '#prompt', '#send', '#undo', '#inspectChanges']) {
      const box = await page.locator(selector).boundingBox()
      expect(box.y).toBeGreaterThanOrEqual(0)
      expect(box.y + box.height).toBeLessThanOrEqual(viewport.height)
      expect(box.x + box.width).toBeLessThanOrEqual(viewport.width)
    }
    expect(await page.evaluate(() => ({ height: document.documentElement.scrollHeight, width: document.documentElement.scrollWidth }))).toEqual({ height: viewport.height, width: viewport.width })
  }
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.locator('#conversation').evaluate(element => {
    const message = document.createElement('p')
    message.textContent = 'A long conversation about live game changes. '.repeat(1000)
    element.append(message)
  })
  await page.locator('#conversation').hover()
  await page.mouse.wheel(0, 500)
  await expect.poll(() => page.locator('#conversation').evaluate(element => element.scrollTop)).toBeGreaterThan(0)
  expect(await page.evaluate(() => scrollY)).toBe(0)
  await page.screenshot({ path: 'printscreens/viewport.png' })
})

test('game arrows stay in the player and textarea arrows remain editing keys', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 })
  await startGame(page)
  const frame = page.frames().find(frame => frame.url().includes('player.html'))
  await frame.evaluate(() => {
    window.arrowInputs = []
    window.addEventListener('keydown', event => { if (event.key.startsWith('Arrow')) window.arrowInputs.push(event.key) })
  })
  await page.locator('#filterToggle').focus()
  await page.keyboard.press('ArrowDown')
  await expect.poll(() => frame.evaluate(() => window.arrowInputs)).toContain('ArrowDown')
  await page.frameLocator('#player').locator('canvas').first().click()
  await page.keyboard.press('ArrowUp')
  await expect.poll(() => frame.evaluate(() => window.arrowInputs)).toContain('ArrowUp')
  expect(await page.evaluate(() => scrollY)).toBe(0)
  expect(await frame.evaluate(() => scrollY)).toBe(0)
  const count = await frame.evaluate(() => window.arrowInputs.length)
  await page.locator('#prompt').fill('First line\nSecond line')
  await page.keyboard.press('ArrowUp')
  expect(await frame.evaluate(() => window.arrowInputs.length)).toBe(count)
})

test('generic 256-write batches change a cartridge without a profile and undo restores it', async ({ page }) => {
  const cheats = [{ label: 'Blue low byte', code: '7E000000', once: true }, { label: 'Blue high byte', code: '7E00017C', once: true }, ...Array.from({ length: 254 }, (_, index) => ({ label: 'Memory batch', code: (0x7e1000 + index).toString(16).toUpperCase() + 'A5', once: true }))]
  await page.route('**/api/status', route => route.fulfill({ json: { available: true } }))
  await page.route('**/api/change', route => route.fulfill({ json: { message: 'Apply the full one-time batch.', unsupported: false, cheats } }))
  const player = await startGame(page)
  const canvas = player.locator('canvas').first()
  const frame = page.frames().find(frame => frame.url().includes('player.html'))
  const counter = () => frame.evaluate(async () => {
    const { snesRam } = await import('/memory.js')
    return snesRam(window.EJS_emulator.gameManager.getState())[0x10]
  })
  const assertRunning = async () => {
    const before = await counter()
    await expect.poll(counter).not.toBe(before)
  }
  await expect.poll(() => screenColor(page, canvas)).toEqual([255, 0, 0, 255])
  await assertRunning()
  await page.locator('#prompt').fill('Apply a one-time memory batch')
  await page.getByRole('button', { name: 'Ask Codex' }).click()
  await expect(page.locator('#conversation')).toContainText('Verified 256 one-time memory writes')
  await expect(page.locator('#changeCount')).toHaveText('00')
  await expect.poll(() => screenColor(page, canvas)).toEqual([0, 0, 255, 255])
  await assertRunning()
  await page.getByRole('button', { name: 'Undo last change' }).click()
  await expect.poll(() => screenColor(page, canvas)).toEqual([255, 0, 0, 255])
  await assertRunning()
})

test('real SNES core applies live WRAM changes and undo restores video', async ({ page }) => {
  await page.route('**/api/status', route => route.fulfill({ json: { available: true, message: 'Controlled agent response for browser testing' } }))
  await page.route('**/api/change', async route => {
    expect(JSON.parse(route.request().postData()).model).toBe('gpt-5.6-terra')
    await route.fulfill({ json: { message: 'Set the color test cartridge backdrop to blue.', unsupported: false, cheats: [{ label: 'Blue low byte', code: '7E000000' }, { label: 'Blue high byte', code: '7E00017C' }] } })
  })
  const player = await startGame(page)
  await expect(page.getByRole('button', { name: 'HD filter off' })).toBeEnabled()
  await page.getByRole('button', { name: 'HD filter off' }).click()
  await expect(page.getByRole('button', { name: 'HD filter on' })).toBeVisible()
  await page.selectOption('#model', 'gpt-5.6-terra')
  const canvas = player.locator('canvas').first()
  await expect.poll(() => screenColor(page, canvas)).toEqual([255, 0, 0, 255])
  await canvas.screenshot({ path: 'printscreens/core-before.png' })
  await page.locator('#prompt').fill('Set the backdrop blue using 7E000000 and 7E00017C.')
  await page.getByRole('button', { name: 'Ask Codex' }).click()
  await expect(page.locator('#changeCount')).toHaveText('02')
  await expect.poll(() => screenColor(page, canvas)).toEqual([0, 0, 255, 255])
  await canvas.screenshot({ path: 'printscreens/core-after.png' })
  await page.screenshot({ path: 'printscreens/live-changes.png', fullPage: true })
  await page.getByRole('button', { name: 'Undo last change' }).click()
  await expect(page.locator('#changeCount')).toHaveText('00')
  await expect.poll(() => screenColor(page, canvas)).toEqual([255, 0, 0, 255])
  await canvas.screenshot({ path: 'printscreens/core-restored.png' })
})

test('unsupported requests and cancellation leave the game unchanged', async ({ page }) => {
  await page.route('**/api/status', route => route.fulfill({ json: { available: true, message: 'Controlled agent response for browser testing' } }))
  await page.route('**/api/change', route => route.fulfill({ json: { message: 'New levels require ROM development.', unsupported: true, cheats: [] } }))
  await startGame(page)
  await page.locator('#prompt').fill('Add a new level')
  await page.getByRole('button', { name: 'Ask Codex' }).click()
  await expect(page.locator('#conversation')).toContainText('No changes applied.')
  await expect(page.locator('#undo')).toBeDisabled()
  await page.unroute('**/api/change')
  let release
  await page.route('**/api/change', async route => {
    await new Promise(resolve => { release = resolve })
    await route.fulfill({ json: { message: 'Late reply', unsupported: false, cheats: [{ label: 'Color', code: '7E000000' }] } }).catch(() => {})
  })
  await page.locator('#prompt').fill('Change color')
  await page.getByRole('button', { name: 'Ask Codex' }).click()
  await expect.poll(() => Boolean(release)).toBe(true)
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  release()
  await expect(page.locator('#conversation')).toContainText('Request cancelled.')
  await expect(page.locator('#changeCount')).toHaveText('00')
})

if (process.env.LIVE_CODEX === '1') {
  test('real Codex through the local SDK changes the running cartridge', async ({ page }) => {
    await startGame(page)
    await page.locator('#prompt').fill('This is my original color test cartridge. Apply these explicit verified WRAM codes: 7E000000 labeled Blue low byte and 7E00017C labeled Blue high byte. Return both as the full active list.')
    await page.getByRole('button', { name: 'Ask Codex' }).click()
    await expect(page.locator('#changeCount')).toHaveText('02', { timeout: 170000 })
    await expect(page.locator('#conversation')).toContainText('Codes sent to the emulator')
    await page.screenshot({ path: 'printscreens/astra-live.png', fullPage: true })
    await page.getByRole('button', { name: 'Undo last change' }).click()
    await expect(page.locator('#changeCount')).toHaveText('00')
  })
}
