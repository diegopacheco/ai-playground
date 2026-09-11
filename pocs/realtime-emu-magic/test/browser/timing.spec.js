import { test, expect } from '@playwright/test'
import { colorRom } from '../color-rom.mjs'

test('emulation keeps full speed when display callbacks drop to 30 Hz', async ({ page }) => {
  await page.addInitScript(() => {
    const requestFrame = window.requestAnimationFrame.bind(window)
    window.requestAnimationFrame = callback => {
      const start = performance.now()
      const tick = time => {
        if (window.limitRefresh && time - start < 30) return requestFrame(tick)
        callback(time)
      }
      return requestFrame(tick)
    }
  })
  await page.goto('/')
  await page.locator('#romInput').setInputFiles({ name: 'timing.sfc', mimeType: 'application/octet-stream', buffer: colorRom() })
  await page.frameLocator('#player').getByText('Start Game', { exact: true }).click()
  await expect(page.locator('#gameStatus')).toHaveText('NOW PLAYING')
  const frame = page.frames().find(frame => frame.url().includes('player.html'))
  const measure = () => frame.evaluate(async () => {
    const manager = window.EJS_emulator.gameManager
    const start = performance.now()
    const first = manager.getFrameNum()
    await new Promise(resolve => setTimeout(resolve, 5000))
    return (manager.getFrameNum() - first) * 1000 / (performance.now() - start)
  })
  process.stdout.write(`Initial FPS: ${await measure()}\n`)
  await frame.evaluate(() => { window.limitRefresh = true })
  const fps = await measure()
  process.stdout.write(`30 Hz display FPS: ${fps}\n`)
  expect(fps).toBeGreaterThan(50)
  expect(fps).toBeLessThan(70)
})
