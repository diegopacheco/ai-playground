#!/usr/bin/env node
import { chromium } from 'playwright'
import { spawn, execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync, readFileSync, existsSync, rmSync, copyFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const targetDir = resolve(process.argv[2] || '.')
const outDir = resolve(process.argv[3] || join(targetDir, 'bug-recording-output'))

const dirs = {
  out: outDir,
  raw: join(outDir, 'raw'),
  videos: join(outDir, 'videos'),
  frames: join(outDir, 'frames'),
  shots: join(outDir, 'screenshots')
}
for (const d of Object.values(dirs)) mkdirSync(d, { recursive: true })

function log(msg) {
  process.stdout.write(`[bug-recording] ${msg}\n`)
}

function detectStack(dir) {
  const pkgPath = join(dir, 'package.json')
  if (!existsSync(pkgPath)) throw new Error(`no package.json in ${dir}`)
  const pkg = JSON.parse(readFileSync(pkgPath, 'utf8'))
  const deps = { ...pkg.dependencies, ...pkg.devDependencies }
  const framework = deps.react && deps['react-dom'] ? 'react' : 'unknown'
  let runtime = 'node'
  let manager = 'npm'
  if (existsSync(join(dir, 'bun.lockb')) || existsSync(join(dir, 'bun.lock'))) {
    runtime = 'bun'
    manager = 'bun'
  } else if (existsSync(join(dir, 'pnpm-lock.yaml'))) {
    manager = 'pnpm'
  } else if (existsSync(join(dir, 'yarn.lock'))) {
    manager = 'yarn'
  }
  const devScript = pkg.scripts?.dev ? 'dev' : pkg.scripts?.start ? 'start' : null
  if (framework !== 'react') throw new Error('target is not a React app (react + react-dom not found)')
  if (!devScript) throw new Error('no dev or start script in package.json')
  return { framework, runtime, manager, devScript }
}

function startApp(dir, stack) {
  const cmd = stack.runtime === 'bun' ? 'bun' : stack.manager
  const args = ['run', stack.devScript]
  const child = spawn(cmd, args, { cwd: dir, env: process.env })
  return new Promise((res, rej) => {
    let buf = ''
    const onData = chunk => {
      buf += chunk.toString()
      const m = buf.match(/https?:\/\/(?:localhost|127\.0\.0\.1):\d+/)
      if (m) res({ child, baseUrl: m[0] })
    }
    child.stdout.on('data', onData)
    child.stderr.on('data', onData)
    child.on('exit', code => rej(new Error(`dev server exited early (code ${code})`)))
    setTimeout(() => rej(new Error('dev server did not report a URL within 60s')), 60000)
  })
}

async function waitReachable(baseUrl) {
  for (let i = 0; i < 30; i++) {
    try {
      const r = await fetch(baseUrl)
      if (r.ok) return
    } catch {}
    await new Promise(r => setTimeout(r, 1000))
  }
  throw new Error(`app not reachable at ${baseUrl}`)
}

async function discoverRoutes(page, baseUrl) {
  await page.goto(baseUrl, { waitUntil: 'networkidle' })
  const hrefs = await page.$$eval('a[href]', as => as.map(a => a.getAttribute('href')))
  const routes = new Set(['/'])
  for (const h of hrefs) if (h && h.startsWith('/') && !h.startsWith('//')) routes.add(h)
  return [...routes]
}

function componentGuess(route, hint) {
  const map = [
    [/card/i, 'ProductCard'],
    [/stepper|qty/i, 'QtyStepper'],
    [/stat|crash/i, 'Stats']
  ]
  for (const [re, name] of map) if (re.test(hint || '')) return name
  const seg = route.replace(/\//g, ' ').trim().split(' ').pop() || 'home'
  return seg.charAt(0).toUpperCase() + seg.slice(1) + 'Page'
}

async function findCssIssues(page) {
  return page.evaluate(() => {
    const out = []
    for (const el of document.querySelectorAll('*')) {
      const cs = getComputedStyle(el)
      const hidden = cs.overflow === 'hidden' || cs.overflowX === 'hidden'
      const clipped = el.scrollWidth > el.clientWidth + 1
      const text = (el.textContent || '').trim()
      if (hidden && cs.whiteSpace === 'nowrap' && cs.textOverflow !== 'ellipsis' && clipped && text.length > 0) {
        out.push({ cls: String(el.className), text: text.slice(0, 80), scrollWidth: el.scrollWidth, clientWidth: el.clientWidth })
      }
    }
    return out
  })
}

async function findFunctionalIssue(page) {
  return page.evaluate(async () => {
    const buttons = [...document.querySelectorAll('button')]
    const incs = buttons.filter(b => {
      const name = ((b.getAttribute('aria-label') || '') + ' ' + (b.textContent || '')).trim()
      return /increase|increment|add|^\+$/i.test(name)
    })
    for (const btn of incs) {
      const scope = btn.closest('.stepper, li, [data-qty]') || document.body
      let readout = scope.querySelector('[data-testid="qty"]')
      if (!readout) {
        readout = [...scope.querySelectorAll('span, strong, b')].find(s => /^\d+$/.test((s.textContent || '').trim()))
      }
      if (!readout) continue
      const before = Number(readout.textContent)
      btn.click()
      await new Promise(r => setTimeout(r, 80))
      const after = Number(readout.textContent)
      if (Number.isFinite(before) && Number.isFinite(after) && after < before) {
        return { control: (btn.getAttribute('aria-label') || btn.textContent || '').trim(), before, after, cls: String(btn.className) }
      }
    }
    return null
  })
}

function toMp4(webm, mp4) {
  execFileSync('ffmpeg', ['-y', '-i', webm, '-c:v', 'libx264', '-crf', '28', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-r', '30', mp4], { stdio: 'ignore' })
}

function reduceSize(mp4) {
  const out = execFileSync('bash', [join(here, 'reduce-size.sh'), mp4]).toString().trim()
  log(out)
}

function sampleFrames(mp4, prefix) {
  execFileSync('ffmpeg', ['-y', '-i', mp4, '-vf', 'fps=1', `${prefix}-%03d.png`], { stdio: 'ignore' })
}

const overlayInit = `
window.__rec = (() => {
  const ensure = () => {
    if (document.getElementById('__rec_cursor')) return
    const cur = document.createElement('div')
    cur.id = '__rec_cursor'
    Object.assign(cur.style, { position: 'fixed', left: '-60px', top: '-60px', width: '24px', height: '24px', marginLeft: '-12px', marginTop: '-12px', borderRadius: '50%', border: '3px solid #ff2d55', background: 'rgba(255,45,85,0.25)', boxShadow: '0 0 0 4px rgba(255,45,85,0.15)', zIndex: '2147483647', pointerEvents: 'none', transition: 'left .35s ease, top .35s ease' })
    document.documentElement.appendChild(cur)
    const cap = document.createElement('div')
    cap.id = '__rec_caption'
    Object.assign(cap.style, { position: 'fixed', left: '50%', bottom: '26px', transform: 'translateX(-50%)', maxWidth: '82%', padding: '11px 20px', borderRadius: '12px', background: 'rgba(17,17,17,0.92)', color: '#fff', font: '600 16px system-ui, sans-serif', lineHeight: '1.35', textAlign: 'center', zIndex: '2147483647', pointerEvents: 'none', opacity: '0', transition: 'opacity .25s' })
    document.documentElement.appendChild(cap)
  }
  return {
    move: (x, y) => { ensure(); const c = document.getElementById('__rec_cursor'); c.style.left = x + 'px'; c.style.top = y + 'px' },
    ping: () => { ensure(); const c = document.getElementById('__rec_cursor'); c.animate([{ transform: 'scale(1)' }, { transform: 'scale(0.55)' }, { transform: 'scale(1)' }], { duration: 320 }) },
    caption: t => { ensure(); const c = document.getElementById('__rec_caption'); c.textContent = t || ''; c.style.opacity = t ? '1' : '0' },
    box: sel => { document.querySelectorAll('.__rec_box').forEach(b => b.remove()); const el = document.querySelector(sel); if (!el) return false; const r = el.getBoundingClientRect(); const b = document.createElement('div'); b.className = '__rec_box'; Object.assign(b.style, { position: 'fixed', left: r.left + 'px', top: r.top + 'px', width: r.width + 'px', height: r.height + 'px', border: '3px solid #ff2d55', borderRadius: '6px', boxShadow: '0 0 0 3px rgba(255,45,85,0.25)', zIndex: '2147483646', pointerEvents: 'none' }); document.documentElement.appendChild(b); return true },
    clear: () => { for (const id of ['__rec_cursor', '__rec_caption']) { const e = document.getElementById(id); if (e) e.remove() } document.querySelectorAll('.__rec_box').forEach(b => b.remove()) }
  }
})()
`

async function caption(page, text) {
  await page.evaluate(t => window.__rec.caption(t), text)
}

async function box(page, selector) {
  await page.evaluate(sel => window.__rec.box(sel), selector)
}

async function moveCursor(page, x, y) {
  await page.evaluate(([px, py]) => window.__rec.move(px, py), [x, y])
  await page.mouse.move(x, y, { steps: 12 })
  await page.waitForTimeout(450)
}

async function pointAndClick(page, locator) {
  const b = await locator.boundingBox()
  if (!b) { await locator.click(); return }
  await moveCursor(page, b.x + b.width / 2, b.y + b.height / 2)
  await page.evaluate(() => window.__rec.ping())
  await page.waitForTimeout(300)
  await locator.click()
  await page.waitForTimeout(700)
}

function bugTargetBox(bug) {
  if (bug.class === 'functional') return '[data-testid="qty"]'
  if (bug.class === 'render') return '.crash, [role="alert"]'
  return '.card-title'
}

async function playStep(page, baseUrl, bug, step) {
  if (step.action === 'navigate') {
    await caption(page, `Step ${step.n}: open ${step.target}`)
    await page.waitForTimeout(700)
    const link = page.locator(`a[href="${step.target}"]`).first()
    if (await link.count()) {
      await pointAndClick(page, link)
    } else {
      await page.goto(baseUrl + step.target, { waitUntil: 'networkidle' })
    }
    await page.waitForTimeout(900)
  } else if (step.action === 'click') {
    await caption(page, `Step ${step.n}: click ${step.target}`)
    await page.waitForTimeout(700)
    const btn = page.locator('button[aria-label="increase"], button:has-text("+")').first()
    await pointAndClick(page, btn)
    await page.waitForTimeout(1000)
  } else {
    await caption(page, `Step ${step.n}: ${step.target}`)
    await box(page, bugTargetBox(bug))
    await page.waitForTimeout(2600)
  }
}

async function recordBug(browser, baseUrl, bug) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: { dir: dirs.raw, size: { width: 1280, height: 720 } }
  })
  await context.addInitScript(overlayInit)
  const page = await context.newPage()
  await page.goto(baseUrl, { waitUntil: 'networkidle' })
  await page.waitForTimeout(500)
  await moveCursor(page, 640, 360)
  await caption(page, `Bug: ${bug.title}`)
  await page.waitForTimeout(1700)
  for (const step of bug.steps) await playStep(page, baseUrl, bug, step)
  await caption(page, '')
  await page.evaluate(() => window.__rec.clear())
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(dirs.shots, `${bug.id}.png`) })
  const video = page.video()
  await context.close()
  const raw = await video.path()
  const mp4 = join(dirs.videos, `${bug.id}.mp4`)
  toMp4(raw, mp4)
  rmSync(raw, { force: true })
  reduceSize(mp4)
  sampleFrames(mp4, join(dirs.frames, bug.id))
  return { video: `videos/${bug.id}.mp4`, screenshot: `screenshots/${bug.id}.png` }
}

async function main() {
  log(`target: ${targetDir}`)
  const stack = detectStack(targetDir)
  log(`stack: react on ${stack.runtime} (${stack.manager} run ${stack.devScript})`)

  log('starting app...')
  const { child, baseUrl } = await startApp(targetDir, stack)
  let server = child
  try {
    await waitReachable(baseUrl)
    log(`app up at ${baseUrl}`)

    const browser = await chromium.launch()
    const huntCtx = await browser.newContext({ viewport: { width: 1280, height: 720 } })
    const page = await huntCtx.newPage()

    const routes = await discoverRoutes(page, baseUrl)
    log(`routes: ${routes.join(', ')}`)

    const bugs = []
    let n = 0

    for (const route of routes) {
      const errors = []
      const onError = e => errors.push(e.message || String(e))
      const onConsole = m => {
        if (m.type() === 'error') errors.push(m.text())
      }
      page.on('pageerror', onError)
      page.on('console', onConsole)
      await page.goto(baseUrl + route, { waitUntil: 'networkidle' })
      await page.waitForTimeout(700)

      const crash = await page.$('.crash, [role="alert"]')
      if (errors.length > 0 || crash) {
        n++
        const raw = errors[0] || 'render error'
        const matched = raw.match(/[A-Za-z]*Error:[^\n]+/)
        const message = (matched ? matched[0] : raw.split('\n')[0]).trim()
        const comp = raw.match(/\bat (\w+) \(.*?src\/.*?\)/)
        bugs.push({
          id: `bug-${n}`,
          class: 'render',
          title: `Component crashes on ${route}: ${message}`,
          page: route,
          component: comp ? comp[1] : componentGuess(route, 'crash'),
          evidence: { console: errors.slice(0, 3).map(e => e.slice(0, 300)), message },
          steps: [
            { n: 1, action: 'navigate', target: route },
            { n: 2, action: 'observe', target: 'error boundary fallback renders instead of content' }
          ]
        })
        log(`render bug on ${route}`)
      }

      const css = await findCssIssues(page)
      if (css.length > 0) {
        n++
        const first = css[0]
        bugs.push({
          id: `bug-${n}`,
          class: 'css',
          title: `Text clipped without ellipsis (${css.length} element(s)) on ${route}`,
          page: route,
          component: componentGuess(route, first.cls),
          evidence: { metrics: css.slice(0, 4) },
          steps: [
            { n: 1, action: 'navigate', target: route },
            { n: 2, action: 'observe', target: `title clipped: scrollWidth ${first.scrollWidth} > clientWidth ${first.clientWidth}, no ellipsis` }
          ]
        })
        log(`css bug on ${route}`)
      }

      const func = await findFunctionalIssue(page)
      if (func) {
        n++
        bugs.push({
          id: `bug-${n}`,
          class: 'functional',
          title: `"${func.control}" control moves value the wrong way (${func.before} -> ${func.after})`,
          page: route,
          component: componentGuess(route, func.cls),
          evidence: { metrics: func },
          steps: [
            { n: 1, action: 'navigate', target: route },
            { n: 2, action: 'click', target: `the "${func.control}" button` },
            { n: 3, action: 'observe', target: `value decreases from ${func.before} to ${func.after}` }
          ]
        })
        log(`functional bug on ${route}`)
      }

      page.off('pageerror', onError)
      page.off('console', onConsole)
    }

    await huntCtx.close()

    log(`recording ${bugs.length} bug(s)...`)
    for (const bug of bugs) {
      const media = await recordBug(browser, baseUrl, bug)
      bug.video = media.video
      bug.screenshot = media.screenshot
    }
    await browser.close()

    const findings = {
      target: { path: targetDir, stack: { framework: stack.framework, runtime: stack.runtime, manager: stack.manager, devScript: stack.devScript, baseUrl } },
      generatedAt: new Date().toISOString(),
      bugs
    }
    writeFileSync(join(dirs.out, 'findings.json'), JSON.stringify(findings, null, 2))
    log(`wrote ${join(dirs.out, 'findings.json')} with ${bugs.length} bug(s)`)
  } finally {
    if (server) server.kill('SIGTERM')
  }
}

main().catch(err => {
  log(`error: ${err.message}`)
  process.exit(1)
})
