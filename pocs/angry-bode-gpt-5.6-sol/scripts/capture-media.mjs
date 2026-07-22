import { execFile } from 'node:child_process'
import { mkdir, unlink } from 'node:fs/promises'
import { resolve } from 'node:path'
import { promisify } from 'node:util'
import { chromium } from 'playwright'

const assets = resolve('assets')
const run = promisify(execFile)
await mkdir(assets, { recursive: true })

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const stillContext = await browser.newContext({ viewport: { width: 960, height: 540 } })
const stillPage = await stillContext.newPage()
await stillPage.goto('http://127.0.0.1:5188')
await stillPage.waitForTimeout(900)
await stillPage.screenshot({ path: resolve(assets, 'init-screen.png') })
await stillContext.close()

const videoContext = await browser.newContext({
  viewport: { width: 960, height: 540 },
  recordVideo: { dir: assets, size: { width: 960, height: 540 } }
})
const page = await videoContext.newPage()
await page.goto('http://127.0.0.1:5188')
await page.getByRole('button', { name: 'UNLEASH THE BODE!' }).click()
const startedAt = Date.now()

await page.keyboard.down('KeyD')
await page.waitForTimeout(650)
await page.keyboard.up('KeyD')
await page.keyboard.down('KeyE')
await page.waitForTimeout(850)
await page.keyboard.up('KeyE')
await page.keyboard.down('ArrowDown')
await page.waitForTimeout(750)
await page.keyboard.up('ArrowDown')
await page.keyboard.press('Space')
await page.waitForTimeout(350)
await page.screenshot({ path: resolve(assets, 'gameplay-screen.png') })
await page.keyboard.down('ArrowRight')
await page.waitForTimeout(600)
await page.keyboard.up('ArrowRight')
await page.keyboard.press('KeyW')
await page.waitForTimeout(350)
await page.keyboard.press('KeyR')
await page.waitForTimeout(500)
await page.keyboard.down('KeyE')
await page.waitForTimeout(600)
await page.keyboard.up('KeyE')
await page.waitForTimeout(Math.max(0, 5000 - (Date.now() - startedAt)))

const video = page.video()
await videoContext.close()
const recordedVideo = await video.path()
const savedVideo = resolve(assets, 'angry-bode-play.webm')
await video.saveAs(savedVideo)
if (recordedVideo !== savedVideo) await unlink(recordedVideo)
await browser.close()
await run('ffmpeg', [
  '-y',
  '-ss', '1.25',
  '-t', '5',
  '-i', savedVideo,
  '-vf', 'fps=12,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
  '-loop', '0',
  resolve(assets, 'angry-bode-play.gif')
])
await unlink(savedVideo)
