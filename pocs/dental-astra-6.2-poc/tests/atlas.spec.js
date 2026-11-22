import { test, expect } from '@playwright/test'
import { teeth } from '../src/teeth.js'
const sequence = [
  'Third molar',
  'Second molar',
  'First molar',
  'Second premolar',
  'First premolar',
  'Canine',
  'Lateral incisor',
  'Central incisor',
]
test('all designations match the four ADA quadrants', () => {
  const ordered = [
    ...sequence,
    ...sequence.toReversed(),
    ...sequence,
    ...sequence.toReversed(),
  ]
  expect(teeth.map((t) => t.name)).toEqual(ordered)
  for (let i = 0; i < 32; i++) {
    expect(teeth[i]).toMatchObject({
      id: i + 1,
      arch: i < 16 ? 'Maxillary' : 'Mandibular',
      side: i < 8 || i >= 24 ? 'Right' : 'Left',
    })
  }
})
test('desktop render, index and keyboard selection survive rotation and zoom', async ({
  page,
}) => {
  const errors = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  page.on('response', (r) => {
    if (r.status() >= 400) errors.push(r.url())
  })
  await page.goto('/')
  const tag = page.locator('.tag.active')
  await expect(tag).toHaveText('8')
  await page.screenshot({ path: 'printscreens/arches.png', fullPage: true })
  for (const t of teeth) {
    await page.locator('[data-id="' + t.id + '"]').click()
    await expect(page.locator('#name')).toHaveText(t.name)
    await expect(tag).toHaveText(String(t.id))
  }
  const incisor = page.locator('[data-id="9"]')
  await incisor.focus()
  await incisor.press('Enter')
  const box = await page.locator('canvas').boundingBox()
  const initial = await tag.boundingBox()
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(
    box.x + box.width / 2 + 65,
    box.y + box.height / 2 + 20,
    { steps: 4 },
  )
  await page.mouse.up()
  await page.mouse.wheel(0, -100)
  await expect
    .poll(async () => (await tag.boundingBox())?.x)
    .not.toEqual(initial.x)
  await expect(tag).toHaveText('9')
  await expect(page.locator('#number')).toHaveText('#9')
  await page.getByLabel('Reset view').click()
  await expect
    .poll(async () => Math.round((await tag.boundingBox()).x))
    .toBe(Math.round(initial.x))
  expect(errors).toEqual([])
})
test('narrow layout and unavailable WebGL', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await page.goto('/')
  await expect(page.locator('.tag.active')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    375,
  )
  await page.screenshot({ path: 'printscreens/mobile.png', fullPage: true })
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      return type.includes('webgl') ? null : original.call(this, type, ...args)
    }
  })
  await page.reload()
  await expect(page.locator('#fallback')).toBeVisible()
  await page.locator('[data-id="32"]').click()
  await expect(page.locator('#number')).toHaveText('#32')
})
test('click the four wisdom crowns on the model in patient orientation', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.locator('.tag.active')).toHaveText('8')
  const canvas = page.locator('canvas')
  const box = await canvas.boundingBox()
  for (const [id, x, y] of [
    [1, 0.322, 0.431],
    [16, 0.646, 0.431],
    [17, 0.633, 0.58],
    [32, 0.332, 0.58],
  ]) {
    await canvas.click({ position: { x: box.width * x, y: box.height * y } })
    await expect(page.locator('#number')).toHaveText('#' + id)
    await expect(page.locator('.tag.active')).toHaveText(String(id))
    await expect(page.locator('[data-id="' + id + '"]')).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  }
  for (const arch of ['Upper', 'Lower']) {
    await page.getByRole('button', { name: arch, exact: true }).click()
    const visible = page.locator('.tag:visible')
    await expect(visible).toHaveCount(16)
    await expect
      .poll(async () =>
        (await visible.allTextContents())
          .map(Number)
          .every((id) => (arch === 'Upper' ? id <= 16 : id > 16)),
      )
      .toBe(true)
    await page.screenshot({
      path: 'printscreens/' + arch.toLowerCase() + '.png',
      fullPage: true,
    })
  }
})
