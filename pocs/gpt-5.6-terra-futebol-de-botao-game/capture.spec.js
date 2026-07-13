const { test } = require('@playwright/test')

test.use({
  browserName: 'chromium',
  channel: 'chrome',
  viewport: { width: 1440, height: 900 }
})

test('capture gameplay', async ({ page }) => {
  await page.goto('http://127.0.0.1:8091/')
  await page.click('[data-mode="human"]')
  await page.click('[data-narration="no"]')
  await page.click('#startButton')
  const canvas = page.locator('#game')
  const box = await canvas.boundingBox()
  const point = (x, y) => ({ x: box.x + x / 1200 * box.width, y: box.y + y / 700 * box.height })
  const start = point(465, 280)
  const pull = point(345, 220)
  await page.mouse.move(start.x, start.y)
  await page.mouse.down()
  await page.mouse.move(pull.x, pull.y, { steps: 8 })
  await page.mouse.up()
  for (let frame = 0; frame < 50; frame += 1) {
    await page.screenshot({ path: `docs/gif-frames/frame-${String(frame).padStart(3, '0')}.png` })
    await page.waitForTimeout(100)
  }
})
