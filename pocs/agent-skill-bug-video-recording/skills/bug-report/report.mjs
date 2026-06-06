#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync, copyFileSync, existsSync, readdirSync, createReadStream, statSync } from 'node:fs'
import { join, resolve, basename, extname } from 'node:path'
import { createServer } from 'node:http'

const args = process.argv.slice(2)
const positional = args.filter(a => !a.startsWith('--'))
const flags = args.filter(a => a.startsWith('--'))
const inDir = resolve(positional[0] || '.')
const siteDir = resolve(positional[1] || join(inDir, '..', 'bug-report-site'))
const wantServe = flags.includes('--serve')
const portFlag = flags.find(a => a.startsWith('--port='))
const port = portFlag ? Number(portFlag.slice('--port='.length)) : 7800

const findingsPath = join(inDir, 'findings.json')
if (!existsSync(findingsPath)) {
  process.stderr.write(`[bug-report] no findings.json in ${inDir}\n`)
  process.exit(1)
}
const findings = JSON.parse(readFileSync(findingsPath, 'utf8'))

const assetsDir = join(siteDir, 'assets')
mkdirSync(assetsDir, { recursive: true })

const videosDir = join(inDir, 'videos')
if (existsSync(videosDir)) {
  for (const f of readdirSync(videosDir)) copyFileSync(join(videosDir, f), join(assetsDir, f))
}
const shotsDir = join(inDir, 'screenshots')
if (existsSync(shotsDir)) {
  for (const f of readdirSync(shotsDir)) copyFileSync(join(shotsDir, f), join(assetsDir, f))
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function classLabel(c) {
  return { css: 'CSS', functional: 'Functional', render: 'Render' }[c] || c
}

function evidenceLine(bug) {
  if (bug.class === 'css') {
    const m = bug.evidence?.metrics?.[0]
    const count = bug.evidence?.metrics?.length || 0
    return m ? `${count} element(s) clipped — e.g. scrollWidth ${m.scrollWidth} > clientWidth ${m.clientWidth}, no ellipsis` : 'clipped text'
  }
  if (bug.class === 'functional') {
    const m = bug.evidence?.metrics
    return m ? `"${m.control}": ${m.before} → ${m.after} (wrong direction)` : 'wrong behavior'
  }
  if (bug.class === 'render') {
    return bug.evidence?.message || bug.title
  }
  return ''
}

const bugs = findings.bugs || []
const counts = bugs.reduce((a, b) => ((a[b.class] = (a[b.class] || 0) + 1), a), {})
const stack = findings.target?.stack || {}
const when = new Date(findings.generatedAt || Date.now()).toLocaleString()

const cards = bugs.map(b => `
        <article class="card ${b.class}">
          <div class="card-head">
            <span class="badge ${b.class}">${classLabel(b.class)}</span>
            <span class="route">${esc(b.page)}</span>
          </div>
          <h3 class="card-title">${esc(b.title)}</h3>
          <div class="meta">
            <span class="chip">component <strong>${esc(b.component)}</strong></span>
          </div>
          <p class="evidence">${esc(evidenceLine(b))}</p>
        </article>`).join('')

const videoList = bugs.map((b, i) => `
          <button class="vid-item ${i === 0 ? 'active' : ''}" data-vid="${esc(b.id)}">
            <span class="badge ${b.class}">${classLabel(b.class)}</span>
            <span class="vid-title">${esc(b.component)} — ${esc(b.page)}</span>
          </button>`).join('')

const videoPanels = bugs.map((b, i) => `
          <figure class="vid-panel ${i === 0 ? 'active' : ''}" data-panel="${esc(b.id)}">
            <video controls preload="metadata" poster="assets/${esc(basename(b.screenshot || ''))}" src="assets/${esc(basename(b.video || ''))}"></video>
            <figcaption>${esc(b.title)}</figcaption>
          </figure>`).join('')

const repro = bugs.map(b => `
        <article class="repro">
          <div class="card-head">
            <span class="badge ${b.class}">${classLabel(b.class)}</span>
            <span class="route">${esc(b.component)} · ${esc(b.page)}</span>
          </div>
          <ol class="steps">
            ${(b.steps || []).map(s => `<li><span class="act">${esc(s.action)}</span> ${esc(s.target)}</li>`).join('')}
          </ol>
        </article>`).join('')

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Bug Report</title>
<style>
:root{--bg:#f5f6fb;--surface:#fff;--ink:#1c2230;--muted:#6b7280;--line:#e8eaf2;--brand:#5b6cff;
--css:#7c4dff;--css-bg:#efeaff;--functional:#d98300;--functional-bg:#fff2dc;--render:#e23b54;--render-bg:#ffe7eb;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
.wrap{max-width:1040px;margin:0 auto;padding:32px 24px 64px}
header.top{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:8px}
h1{margin:0;font-size:30px;letter-spacing:-0.02em}
.sub{color:var(--muted);font-size:14px;margin-top:6px}
.summary{display:flex;gap:10px;flex-wrap:wrap}
.pill{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:8px 14px;font-weight:600;font-size:13px;box-shadow:0 4px 14px rgba(28,34,48,.04)}
.dot{width:9px;height:9px;border-radius:50%}
.dot.css{background:var(--css)}.dot.functional{background:var(--functional)}.dot.render{background:var(--render)}
nav.tabs{display:flex;gap:6px;margin:26px 0 22px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:6px;width:fit-content;box-shadow:0 6px 18px rgba(28,34,48,.05)}
.tab{border:none;background:transparent;color:var(--muted);font-weight:600;font-size:14px;padding:10px 18px;border-radius:10px;cursor:pointer}
.tab.active{background:var(--brand);color:#fff}
.panel{display:none}.panel.active{display:block}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px;box-shadow:0 8px 22px rgba(28,34,48,.05);border-top:4px solid var(--line)}
.card.css{border-top-color:var(--css)}.card.functional{border-top-color:var(--functional)}.card.render{border-top-color:var(--render)}
.card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.badge{font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:4px 10px;border-radius:999px}
.badge.css{color:var(--css);background:var(--css-bg)}.badge.functional{color:var(--functional);background:var(--functional-bg)}.badge.render{color:var(--render);background:var(--render-bg)}
.route{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted);background:var(--bg);padding:3px 9px;border-radius:7px}
.card-title{margin:2px 0 12px;font-size:16px;line-height:1.4}
.meta{display:flex;gap:8px;margin-bottom:12px}
.chip{font-size:12px;color:var(--muted);background:var(--bg);border:1px solid var(--line);padding:4px 10px;border-radius:8px}
.evidence{margin:0;font-size:13px;color:var(--muted);line-height:1.5}
.video-layout{display:grid;grid-template-columns:260px 1fr;gap:18px}
.vid-list{display:flex;flex-direction:column;gap:8px}
.vid-item{display:flex;flex-direction:column;gap:6px;align-items:flex-start;text-align:left;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px;cursor:pointer}
.vid-item.active{border-color:var(--brand);box-shadow:0 0 0 2px rgba(91,108,255,.15)}
.vid-title{font-size:13px;font-weight:600}
.vid-stage{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 8px 22px rgba(28,34,48,.05)}
.vid-panel{display:none;margin:0}.vid-panel.active{display:block}
.vid-panel video{width:100%;border-radius:12px;background:#000;display:block}
.vid-panel figcaption{margin-top:12px;font-size:13px;color:var(--muted)}
.repro{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:14px;box-shadow:0 8px 22px rgba(28,34,48,.05)}
.steps{margin:12px 0 0;padding-left:20px;line-height:1.9}
.steps .act{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--brand);background:#eef0ff;padding:2px 7px;border-radius:6px;margin-right:4px}
.note{font-size:12px;color:var(--muted);margin:14px 2px 0}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>Bug Report</h1>
      <div class="sub">${esc(stack.framework || 'app')} on ${esc(stack.runtime || 'node')} · ${esc(findings.target?.path || '')} · ${esc(when)}</div>
    </div>
    <div class="summary">
      <span class="pill"><span class="dot css"></span>${counts.css || 0} CSS</span>
      <span class="pill"><span class="dot functional"></span>${counts.functional || 0} Functional</span>
      <span class="pill"><span class="dot render"></span>${counts.render || 0} Render</span>
    </div>
  </header>

  <nav class="tabs">
    <button class="tab active" data-tab="bugs">Bugs Found</button>
    <button class="tab" data-tab="videos">Videos</button>
    <button class="tab" data-tab="repro">Reproduction Steps</button>
  </nav>

  <section class="panel active" data-panel="bugs">
    <div class="grid">${cards}
    </div>
  </section>

  <section class="panel" data-panel="videos">
    <div class="video-layout">
      <div class="vid-list">${videoList}
      </div>
      <div class="vid-stage">${videoPanels}
      </div>
    </div>
  </section>

  <section class="panel" data-panel="repro">
    ${repro}
    <p class="note">Steps are reconstructed from the recorder's captured action log and ffmpeg-sampled frames, not from playing the video file.</p>
  </section>
</div>
<script>
const tabs=document.querySelectorAll('.tab');
const panels=document.querySelectorAll('.panel');
tabs.forEach(t=>t.addEventListener('click',()=>{
  tabs.forEach(x=>x.classList.remove('active'));
  panels.forEach(p=>p.classList.remove('active'));
  t.classList.add('active');
  document.querySelector('.panel[data-panel="'+t.dataset.tab+'"]').classList.add('active');
}));
const items=document.querySelectorAll('.vid-item');
const vpanels=document.querySelectorAll('.vid-panel');
items.forEach(it=>it.addEventListener('click',()=>{
  items.forEach(x=>x.classList.remove('active'));
  vpanels.forEach(p=>{p.classList.remove('active');const v=p.querySelector('video');if(v)v.pause();});
  it.classList.add('active');
  document.querySelector('.vid-panel[data-panel="'+it.dataset.vid+'"]').classList.add('active');
}));
</script>
</body>
</html>`

writeFileSync(join(siteDir, 'index.html'), html)
process.stdout.write(`[bug-report] wrote ${join(siteDir, 'index.html')} (${bugs.length} bug(s))\n`)

if (wantServe) {
  const types = { '.html': 'text/html; charset=utf-8', '.mp4': 'video/mp4', '.png': 'image/png', '.json': 'application/json', '.css': 'text/css', '.js': 'text/javascript', '.svg': 'image/svg+xml' }
  createServer((req, res) => {
    let path = decodeURIComponent(req.url.split('?')[0])
    if (path === '/') path = '/index.html'
    const file = join(siteDir, path)
    if (!file.startsWith(siteDir) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404, { 'content-type': 'text/plain' })
      res.end('not found')
      return
    }
    const type = types[extname(file)] || 'application/octet-stream'
    const size = statSync(file).size
    const range = req.headers.range
    if (range) {
      const match = /bytes=(\d*)-(\d*)/.exec(range) || []
      const start = match[1] ? parseInt(match[1], 10) : 0
      const end = match[2] ? parseInt(match[2], 10) : size - 1
      res.writeHead(206, { 'content-type': type, 'accept-ranges': 'bytes', 'content-range': `bytes ${start}-${end}/${size}`, 'content-length': end - start + 1 })
      createReadStream(file, { start, end }).pipe(res)
    } else {
      res.writeHead(200, { 'content-type': type, 'accept-ranges': 'bytes', 'content-length': size })
      createReadStream(file).pipe(res)
    }
  }).listen(port, () => process.stdout.write(`[bug-report] serving ${siteDir} at http://localhost:${port}/\n`))
}
