import { readFileSync, readdirSync, existsSync, mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'

const ROOT = process.cwd()

type Status = 'pass' | 'fail' | 'missing'

type Section = {
  id: string
  title: string
  source: string
  status: Status
  total: number
  passed: number
  failed: number
  skipped: number
  duration: number
  detail: string
}

function missing(id: string, title: string, where: string): Section {
  return {
    id, title, source: where, status: 'missing',
    total: 0, passed: 0, failed: 0, skipped: 0,
    duration: 0, detail: 'no results found — run tests.sh first'
  }
}

function attr(xml: string, name: string): number {
  const m = xml.match(new RegExp(`<testsuite[^>]*\\b${name}="([\\d.]+)"`))
  return m ? Number(m[1]) : 0
}

function parseSurefire(): Section {
  const dir = join(ROOT, 'backend/target/surefire-reports')
  if (!existsSync(dir)) {
    return missing('junit', 'Backend (JUnit 5: unit + integration)', dir)
  }
  const files = readdirSync(dir).filter(f => f.startsWith('TEST-') && f.endsWith('.xml'))
  if (files.length === 0) {
    return missing('junit', 'Backend (JUnit 5: unit + integration)', dir)
  }
  let tests = 0, failures = 0, errors = 0, skipped = 0, time = 0
  const suiteNames: string[] = []
  for (const f of files) {
    const xml = readFileSync(join(dir, f), 'utf8')
    tests   += attr(xml, 'tests')
    failures += attr(xml, 'failures')
    errors  += attr(xml, 'errors')
    skipped += attr(xml, 'skipped')
    time    += attr(xml, 'time')
    const nameMatch = xml.match(/<testsuite[^>]*\bname="([^"]+)"/)
    if (nameMatch) suiteNames.push(nameMatch[1].split('.').pop()!)
  }
  const failed = failures + errors
  return {
    id: 'junit',
    title: 'Backend (JUnit 5: unit + integration)',
    source: dir,
    status: failed > 0 ? 'fail' : 'pass',
    total: tests,
    passed: tests - failed - skipped,
    failed,
    skipped,
    duration: time,
    detail: `${files.length} suites · ${suiteNames.join(', ')}`
  }
}

function parseJest(): Section {
  const file = join(ROOT, 'frontend/test-results/jest.json')
  if (!existsSync(file)) return missing('jest', 'Frontend (Jest)', file)
  const data = JSON.parse(readFileSync(file, 'utf8'))
  const start = data.startTime as number
  const end = (data.testResults ?? []).reduce(
    (m: number, r: { perfStats?: { end?: number } }) => Math.max(m, r.perfStats?.end ?? 0),
    start
  )
  return {
    id: 'jest',
    title: 'Frontend (Jest)',
    source: file,
    status: data.success ? 'pass' : 'fail',
    total: data.numTotalTests ?? 0,
    passed: data.numPassedTests ?? 0,
    failed: data.numFailedTests ?? 0,
    skipped: (data.numPendingTests ?? 0) + (data.numTodoTests ?? 0),
    duration: end > start ? (end - start) / 1000 : 0,
    detail: `${data.numTotalTestSuites ?? 0} suites`
  }
}

function parsePlaywright(): Section {
  const file = join(ROOT, 'e2e/test-results/playwright.json')
  if (!existsSync(file)) return missing('playwright', 'E2E (Playwright)', file)
  const data = JSON.parse(readFileSync(file, 'utf8'))
  const s = data.stats ?? {}
  const expected = s.expected ?? 0
  const unexpected = s.unexpected ?? 0
  const skipped = s.skipped ?? 0
  const flaky = s.flaky ?? 0
  return {
    id: 'playwright',
    title: 'E2E (Playwright)',
    source: file,
    status: unexpected > 0 ? 'fail' : 'pass',
    total: expected + unexpected + skipped + flaky,
    passed: expected,
    failed: unexpected,
    skipped,
    duration: (s.duration ?? 0) / 1000,
    detail: `flaky=${flaky}`
  }
}

function fmtDuration(ms: number): string {
  if (ms <= 0) return '0ms'
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`
  if (ms < 1000) return `${ms.toFixed(2)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function parseK6(): Section {
  const file = join(ROOT, '.run/k6-summary.json')
  if (!existsSync(file)) return missing('k6', 'Stress (k6)', file)
  const data = JSON.parse(readFileSync(file, 'utf8'))
  const m = data.metrics ?? {}
  const reqs = m.http_reqs?.count ?? 0
  const failRate = m.http_req_failed?.value ?? m.http_req_failed?.rate ?? 0
  const p95 = m.http_req_duration?.['p(95)'] ?? 0
  const avg = m.http_req_duration?.avg ?? 0
  const fails = m.http_req_failed?.passes ?? Math.round(reqs * failRate)
  const passes = reqs - fails
  const checksPasses = m.checks?.passes ?? 0
  const checksFails = m.checks?.fails ?? 0
  return {
    id: 'k6',
    title: 'Stress (k6)',
    source: file,
    status: failRate < 0.01 && checksFails === 0 ? 'pass' : 'fail',
    total: reqs,
    passed: passes,
    failed: fails,
    skipped: 0,
    duration: 0,
    detail: `p95=${fmtDuration(p95)} · avg=${fmtDuration(avg)} · failRate=${(failRate * 100).toFixed(2)}% · checks=${checksPasses}/${checksPasses + checksFails}`
  }
}

const sections = [parseSurefire(), parseJest(), parsePlaywright(), parseK6()]

const totals = sections.reduce(
  (acc, s) => ({
    total:   acc.total   + s.total,
    passed:  acc.passed  + s.passed,
    failed:  acc.failed  + s.failed,
    skipped: acc.skipped + s.skipped
  }),
  { total: 0, passed: 0, failed: 0, skipped: 0 }
)

const anyFail = sections.some(s => s.status === 'fail')
const anyMissing = sections.some(s => s.status === 'missing')
const overall: Status = anyFail ? 'fail' : anyMissing ? 'missing' : 'pass'
const overallLabel = anyFail ? 'FAIL' : anyMissing ? 'PARTIAL' : 'PASS'

function badge(status: Status, label?: string): string {
  const text = label ?? status.toUpperCase()
  return `<span class="badge ${status}">${text}</span>`
}

function card(s: Section): string {
  const passPct = s.total > 0 ? Math.round((s.passed / s.total) * 100) : 0
  return `
    <div class="card ${s.status}">
      <header>
        <h2>${s.title}</h2>
        ${badge(s.status)}
      </header>
      <div class="bar"><div class="bar-fill" style="width:${passPct}%"></div></div>
      <div class="row"><span>Total</span><strong>${s.total}</strong></div>
      <div class="row pass-row"><span>Passed</span><strong>${s.passed}</strong></div>
      <div class="row fail-row"><span>Failed</span><strong>${s.failed}</strong></div>
      <div class="row skip-row"><span>Skipped</span><strong>${s.skipped}</strong></div>
      ${s.duration > 0 ? `<div class="row"><span>Duration</span><strong>${s.duration.toFixed(2)}s</strong></div>` : ''}
      <div class="detail">${s.detail}</div>
      <div class="src">${s.source.replace(ROOT, '.')}</div>
    </div>
  `
}

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Loan App — Test Report</title>
<style>
  :root { font-family: system-ui, -apple-system, sans-serif; color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 32px; background: radial-gradient(circle at 20% 10%, #1f2937, #0b0f17 70%); color: #e5e7eb; min-height: 100vh; }
  h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: 0.3px; }
  .sub { color: #9ca3af; margin: 0 0 18px; font-size: 13px; }
  .overall { display: inline-block; padding: 8px 18px; border-radius: 999px; font-weight: 700; letter-spacing: 1.5px; font-size: 14px; }
  .overall.pass    { background: rgba(52,211,153,0.15);  color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
  .overall.fail    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }
  .overall.missing { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
  .totals { display: flex; gap: 12px; margin: 20px 0 28px; flex-wrap: wrap; }
  .pill { padding: 8px 14px; border-radius: 999px; font-size: 13px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.04); }
  .pill strong { font-variant-numeric: tabular-nums; margin-left: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
  .card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px; }
  .card.pass    { border-color: rgba(52,211,153,0.35); }
  .card.fail    { border-color: rgba(248,113,113,0.45); }
  .card.missing { border-color: rgba(251,191,36,0.35); }
  .card header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
  .card h2 { font-size: 15px; margin: 0; font-weight: 600; }
  .bar { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; margin-bottom: 14px; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #34d399, #60a5fa); transition: width 0.3s; }
  .card.fail .bar-fill { background: linear-gradient(90deg, #f87171, #fbbf24); }
  .card.missing .bar-fill { background: #6b7280; }
  .badge { padding: 3px 9px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
  .badge.pass    { background: rgba(52,211,153,0.18);  color: #34d399; }
  .badge.fail    { background: rgba(248,113,113,0.18); color: #f87171; }
  .badge.missing { background: rgba(251,191,36,0.18);  color: #fbbf24; }
  .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; }
  .row strong { font-variant-numeric: tabular-nums; }
  .pass-row strong { color: #34d399; }
  .fail-row strong { color: #f87171; }
  .skip-row strong { color: #fbbf24; }
  .detail { margin-top: 12px; font-size: 12px; color: #9ca3af; line-height: 1.5; }
  .src { margin-top: 6px; font-size: 11px; color: #6b7280; font-family: ui-monospace, SFMono-Regular, monospace; word-break: break-all; }
  footer { margin-top: 36px; font-size: 12px; color: #6b7280; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 16px; }
</style>
</head>
<body>
  <h1>Loan App — Test Report</h1>
  <p class="sub">Generated ${new Date().toISOString()}</p>
  <div class="overall ${overall}">${overallLabel}</div>
  <div class="totals">
    <span class="pill">Total <strong>${totals.total}</strong></span>
    <span class="pill" style="color:#34d399">Passed <strong>${totals.passed}</strong></span>
    <span class="pill" style="color:#f87171">Failed <strong>${totals.failed}</strong></span>
    <span class="pill" style="color:#fbbf24">Skipped <strong>${totals.skipped}</strong></span>
  </div>
  <div class="grid">
    ${sections.map(card).join('')}
  </div>
  <footer>
    Sources aggregated: JUnit XML (Maven Surefire) · Jest JSON · Playwright JSON · k6 summary JSON.<br/>
    Re-run via <code>./tests.sh</code>, then refresh this page via <code>./report.sh</code>.
  </footer>
</body>
</html>`

mkdirSync(join(ROOT, 'report'), { recursive: true })
writeFileSync(join(ROOT, 'report/index.html'), html)
console.log(`report written: report/index.html  (overall: ${overallLabel})`)
