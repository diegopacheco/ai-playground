#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'

const HERE = dirname(fileURLToPath(import.meta.url))
const TEMPLATE = join(HERE, '..', 'assets', 'template.html')

const KEYWORDS = new Set(['abstract', 'and', 'as', 'assert', 'async', 'await', 'boolean', 'break', 'case', 'catch', 'class', 'const', 'constructor', 'continue', 'crate', 'def', 'default', 'del', 'delete', 'do', 'elif', 'else', 'enum', 'export', 'extends', 'extern', 'false', 'final', 'finally', 'fn', 'for', 'from', 'func', 'function', 'go', 'if', 'impl', 'implements', 'import', 'in', 'instanceof', 'int', 'interface', 'is', 'lambda', 'let', 'let', 'match', 'mod', 'module', 'mut', 'new', 'nil', 'none', 'not', 'null', 'or', 'package', 'pass', 'private', 'protected', 'public', 'pub', 'raise', 'range', 'record', 'return', 'select', 'self', 'static', 'struct', 'super', 'switch', 'this', 'throw', 'throws', 'trait', 'true', 'try', 'type', 'typeof', 'undefined', 'use', 'val', 'var', 'void', 'when', 'where', 'while', 'with', 'yield'])

const TOKENS = /(\/\*[\s\S]*?\*\/|"""[\s\S]*?"""|<!--[\s\S]*?-->)|(\/\/[^\n]*|#[^\n]*|--[^\n]*)|(`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?\b)|(@[A-Za-z_][\w.]*)|([A-Za-z_][\w$]*)/g

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

const highlight = (code) => code.replace(TOKENS, (m, block, line, str, num, ann, word) => {
  if (block || line) return `<span class="t-com">${esc(m)}</span>`
  if (str) return `<span class="t-str">${esc(m)}</span>`
  if (num) return `<span class="t-num">${esc(m)}</span>`
  if (ann) return `<span class="t-ann">${esc(m)}</span>`
  if (word && KEYWORDS.has(word)) return `<span class="t-key">${esc(m)}</span>`
  return esc(m)
})

const diffLine = (line) => {
  const cls = line.startsWith('+++') || line.startsWith('---') ? 'd-head' : line.startsWith('+') ? 'd-add' : line.startsWith('-') ? 'd-del' : line.startsWith('@@') ? 'd-hunk' : ''
  return cls ? `<span class="${cls}">${esc(line)}</span>` : esc(line)
}

const codeBlock = (code, lang, startLine) => {
  const text = String(code ?? '').replace(/\n+$/, '')
  const start = Number.isFinite(startLine) && startLine > 0 ? startLine : 1
  const lines = text.split('\n')
  const gutter = lines.map((_, i) => start + i).join('\n')
  const src = lang === 'diff' ? lines.map(diffLine).join('\n') : highlight(text)
  return `<div class="code"><div class="code-head"><span class="lang">${esc(lang || 'text')}</span><span class="lines">${lines.length} lines</span></div><div class="code-body"><pre class="gutter">${gutter}</pre><pre class="src">${src}</pre></div></div>`
}

const bullets = (items) => `<ul class="list">${(items || []).map((i) => `<li>${esc(i)}</li>`).join('')}</ul>`

const slug = (s) => String(s || 'bug').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'bug'

const stamp = () => {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

const need = (data, path) => {
  const value = path.split('.').reduce((acc, k) => (acc == null ? acc : acc[k]), data)
  if (value == null || value === '' || (Array.isArray(value) && value.length === 0)) {
    console.error(`ERROR: triage json is missing required field: ${path}`)
    process.exit(1)
  }
  return value
}

const input = process.argv[2]
if (!input) {
  console.error('usage: render.mjs <triage.json> [outdir]')
  process.exit(1)
}

const data = JSON.parse(readFileSync(input, 'utf8'))
need(data, 'bug.name')
need(data, 'description')
need(data, 'files')
need(data, 'repro.code')
need(data, 'why_bad')
need(data, 'solution.summary')
need(data, 'files_to_touch')
need(data, 'breaking.verdict')
need(data, 'safety.verdict')

const outDir = resolve(process.argv[3] || join(tmpdir(), `bug-triage-${slug(data.bug.id || data.bug.name)}-${stamp()}`))
mkdirSync(outDir, { recursive: true })

const bug = data.bug
const severity = (bug.severity || 'unknown').toLowerCase()
const trackerLink = bug.url
  ? `<a class="tracker" href="${esc(bug.url)}" target="_blank" rel="noreferrer"><span class="dot"></span>${esc(bug.tracker || 'tracker')}${bug.id ? ` · ${esc(bug.id)}` : ''}</a>`
  : `<span class="tracker none"><span class="dot"></span>no tracker link · reported as description</span>`

const verdictClass = (v) => {
  const t = String(v).toLowerCase()
  if (t.startsWith('no') || t.startsWith('safe')) return 'good'
  if (t.startsWith('yes') || t.startsWith('unsafe')) return 'bad'
  return 'warn'
}

const fileRows = (data.files || []).map((f, i) => `<tr><td class="idx">${i + 1}</td><td class="path"><code>${esc(f.path)}</code>${f.lines ? `<span class="range">:${esc(f.lines)}</span>` : ''}</td><td class="role">${esc(f.role || '')}</td></tr>`).join('')

const touchRows = (data.files_to_touch || []).map((f) => `<tr><td class="path"><code>${esc(f.path)}</code></td><td class="role">${esc(f.change || '')}</td></tr>`).join('')

const sections = [
  { n: 1, id: 'name', title: 'Bug name', body: `<h3 class="bugname">${esc(bug.name)}</h3><div class="meta-row">${trackerLink}<span class="chip sev-${esc(severity)}">severity ${esc(severity)}</span>${bug.branch ? `<span class="chip">branch ${esc(bug.branch)}</span>` : ''}${bug.repo ? `<span class="chip">${esc(bug.repo)}</span>` : ''}</div>` },
  { n: 2, id: 'description', title: 'What the bug is', body: `<div class="prose">${(data.description || []).map((l) => `<p>${esc(l)}</p>`).join('')}</div>` },
  { n: 3, id: 'files', title: 'Files and classes involved', body: `<table class="tbl"><thead><tr><th>#</th><th>Full path</th><th>Why it is involved</th></tr></thead><tbody>${fileRows}</tbody></table>` },
  { n: 4, id: 'repro', title: 'Test that reproduces it', body: `${data.repro.path ? `<p class="hint">Save as <code>${esc(data.repro.path)}</code>${data.repro.run ? ` · run <code>${esc(data.repro.run)}</code>` : ''}</p>` : ''}${codeBlock(data.repro.code, data.repro.language, data.repro.start_line)}${data.repro.expectation ? `<p class="hint">${esc(data.repro.expectation)}</p>` : ''}` },
  { n: 5, id: 'why', title: 'Why this bug is bad', body: bullets(data.why_bad) },
  { n: 6, id: 'solution', title: 'Minimal solution', body: `<div class="prose"><p>${esc(data.solution.summary)}</p></div>${data.solution.code ? codeBlock(data.solution.code, data.solution.language || 'diff', data.solution.start_line) : ''}${data.solution.notes ? bullets(data.solution.notes) : ''}` },
  { n: 7, id: 'touch', title: 'Files to touch', body: `<table class="tbl"><thead><tr><th>Full path</th><th>Change</th></tr></thead><tbody>${touchRows}</tbody></table>` },
  { n: 8, id: 'breaking', title: 'Breaking change?', body: `<div class="verdict ${verdictClass(data.breaking.verdict)}"><span class="v-label">${esc(data.breaking.verdict)}</span><div class="v-flags"><span class="flag ${data.breaking.db ? 'on' : 'off'}">database schema</span><span class="flag ${data.breaking.api ? 'on' : 'off'}">public API / contract</span><span class="flag ${data.breaking.consumers ? 'on' : 'off'}">breaks consumers</span></div></div><div class="prose"><p>${esc(data.breaking.detail || '')}</p></div>` },
  { n: 9, id: 'safety', title: 'Is the fix safe?', body: `<div class="verdict ${verdictClass(data.safety.verdict)}"><span class="v-label">${esc(data.safety.verdict)}</span></div><div class="prose"><p>${esc(data.safety.detail || '')}</p></div>${data.safety.risks ? `<h4>Risks</h4>${bullets(data.safety.risks)}` : ''}${data.safety.verification ? `<h4>How to verify</h4>${bullets(data.safety.verification)}` : ''}` },
]

const nav = sections.map((s) => `<a href="#${s.id}"><span class="n">${s.n}</span>${esc(s.title)}</a>`).join('')
const body = sections.map((s) => `<section id="${s.id}" class="card"><header><span class="num">${s.n}</span><h2>${esc(s.title)}</h2></header>${s.body}</section>`).join('')

const html = readFileSync(TEMPLATE, 'utf8')
  .replace(/{{TITLE}}/g, esc(bug.name))
  .replace('{{SUBTITLE}}', esc(bug.id ? `${bug.id} · bug triage report` : 'bug triage report'))
  .replace('{{GENERATED}}', esc(bug.generatedAt || new Date().toISOString().replace('T', ' ').slice(0, 16)))
  .replace('{{NAV}}', nav)
  .replace('{{SECTIONS}}', body)

const out = join(outDir, 'index.html')
writeFileSync(out, html)
writeFileSync(join(outDir, 'triage.json'), JSON.stringify(data, null, 2))

console.log(`bug triage  ${bug.name}`)
console.log(`  tracker ${bug.url || 'none'}`)
console.log(`  files ${(data.files || []).length}  to touch ${(data.files_to_touch || []).length}`)
console.log(`  breaking ${data.breaking.verdict}  safe ${data.safety.verdict}`)
console.log(`REPORT ${out}`)
