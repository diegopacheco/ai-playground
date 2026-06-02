#!/usr/bin/env node
import { execSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';
import { tmpdir } from 'node:os';

const SLOW_TEST_SECONDS = 5;
const CC_THRESHOLD = 10;
const FUNC_LEN_THRESHOLD = 60;
const FILE_LOC_THRESHOLD = 400;
const SKIP_DIRS = new Set(['node_modules', 'target', 'dist', 'build', '.git', '.lint', '.idea', '.vite']);

const root = process.argv[2] ? resolve(process.argv[2]) : process.cwd();

function resolve(p) {
  return p.startsWith('/') ? p : join(process.cwd(), p);
}

function run(cmd, cwd) {
  const start = Date.now();
  try {
    execSync(cmd, { cwd, stdio: 'pipe', encoding: 'utf8', timeout: 300000 });
    return { ok: true, ms: Date.now() - start };
  } catch (err) {
    return { ok: false, ms: Date.now() - start, output: String(err.stdout || '') + String(err.stderr || '') };
  }
}

function listDir(dir) {
  try {
    return readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

function discoverModules(base) {
  const mods = [];
  (function rec(dir, depth) {
    if (depth > 4) return;
    const entries = listDir(dir);
    const names = entries.map(e => e.name);
    if (names.includes('pom.xml')) {
      mods.push({ dir, type: 'java', build: 'maven' });
    } else if (names.includes('build.gradle') || names.includes('build.gradle.kts')) {
      mods.push({ dir, type: 'java', build: 'gradle' });
    }
    if (names.includes('package.json')) {
      mods.push({ dir, type: 'node' });
    }
    for (const e of entries) {
      if (e.isDirectory() && !SKIP_DIRS.has(e.name)) rec(join(dir, e.name), depth + 1);
    }
  })(base, 0);
  return mods;
}

function walkFiles(dir, exts) {
  const out = [];
  (function rec(d) {
    for (const e of listDir(d)) {
      if (e.isDirectory()) {
        if (!SKIP_DIRS.has(e.name)) rec(join(d, e.name));
      } else if (exts.some(x => e.name.endsWith(x))) {
        out.push(join(d, e.name));
      }
    }
  })(dir);
  return out;
}

function lineOf(content, index) {
  let line = 1;
  for (let i = 0; i < index && i < content.length; i++) {
    if (content[i] === '\n') line++;
  }
  return line;
}

function decisionPoints(text) {
  const keywords = (text.match(/\b(if|for|while|case|catch)\b/g) || []).length;
  const logical = (text.match(/&&|\|\|/g) || []).length;
  const ternary = (text.match(/\?(?![.?:])/g) || []).length;
  return keywords + logical + ternary;
}

function findFunctions(content) {
  const blocks = [];
  const len = content.length;
  for (let i = 0; i < len; i++) {
    if (content[i] !== '{') continue;
    let j = i - 1;
    while (j >= 0 && /\s/.test(content[j])) j--;
    if (content[j] !== ')') continue;
    let depth = 0;
    let k = j;
    for (; k >= 0; k--) {
      if (content[k] === ')') depth++;
      else if (content[k] === '(') {
        depth--;
        if (depth === 0) break;
      }
    }
    if (k < 0) continue;
    let m = k - 1;
    while (m >= 0 && /\s/.test(content[m])) m--;
    const end = m;
    while (m >= 0 && /[\w$]/.test(content[m])) m--;
    const name = content.slice(m + 1, end + 1);
    if (!name || /^(if|for|while|switch|catch|return|function)$/.test(name)) continue;
    let bd = 0;
    let p = i;
    for (; p < len; p++) {
      if (content[p] === '{') bd++;
      else if (content[p] === '}') {
        bd--;
        if (bd === 0) break;
      }
    }
    const body = content.slice(i, p + 1);
    blocks.push({ name, start: i, body });
  }
  return blocks;
}

function parseJUnit(xml) {
  const cases = [];
  const re = /<testcase\b([^>]*?)(\/>|>([\s\S]*?)<\/testcase>)/g;
  let m;
  while ((m = re.exec(xml))) {
    const attrs = m[1];
    const inner = m[3] || '';
    const name = (attrs.match(/name="([^"]*)"/) || [])[1] || '';
    const cls = (attrs.match(/classname="([^"]*)"/) || [])[1] || '';
    const time = parseFloat((attrs.match(/time="([^"]*)"/) || [])[1] || '0');
    const failed = /<failure|<error/.test(inner);
    const skipped = /<skipped/.test(inner);
    cases.push({
      name,
      classname: cls,
      durationMs: Math.round(time * 1000),
      status: skipped ? 'skipped' : failed ? 'failed' : 'passed',
      slow: time >= SLOW_TEST_SECONDS
    });
  }
  return cases;
}

function readSurefire(moduleDir) {
  const reportsDir = join(moduleDir, 'target', 'surefire-reports');
  const files = listDir(reportsDir).filter(e => e.name.endsWith('.xml'));
  let cases = [];
  for (const f of files) {
    cases = cases.concat(parseJUnit(readFileSync(join(reportsDir, f.name), 'utf8')));
  }
  return cases;
}

function buildAndTestJava(mod) {
  const build = run('mvn -q -DskipTests package', mod.dir);
  const test = run('mvn -q test', mod.dir);
  const cases = readSurefire(mod.dir);
  return { build, testRan: true, cases };
}

function buildAndTestNode(mod) {
  const pkg = JSON.parse(readFileSync(join(mod.dir, 'package.json'), 'utf8'));
  const scripts = pkg.scripts || {};
  const deps = { ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) };
  const hasDeps = Object.keys(deps).length > 0;
  if (hasDeps && !existsSync(join(mod.dir, 'node_modules'))) {
    run('bun install', mod.dir);
  }
  let build = { ok: true, ms: 0, skipped: true };
  if (scripts.build) {
    build = run('bun run build', mod.dir);
  }
  let cases = [];
  let testRan = false;
  const testFiles = walkFiles(mod.dir, ['.test.js', '.test.mjs', '.spec.js']);
  const usesNodeTest = (scripts.test || '').includes('node --test')
    || (testFiles.length > 0 && !deps.vitest && !deps.jest && !deps.mocha);
  if (usesNodeTest && testFiles.length > 0) {
    const dest = join(tmpdir(), 'lint-junit-' + Date.now() + '.xml');
    run('node --test --test-reporter=junit --test-reporter-destination=' + dest, mod.dir);
    if (existsSync(dest)) {
      cases = parseJUnit(readFileSync(dest, 'utf8'));
      testRan = true;
    }
  }
  return { build, testRan, cases };
}

function analyzeComplexity(mod) {
  const exts = mod.type === 'java' ? ['.java'] : ['.js', '.jsx', '.ts', '.tsx', '.mjs'];
  const files = [];
  for (const f of walkFiles(mod.dir, exts)) {
    const content = readFileSync(f, 'utf8');
    const functions = findFunctions(content).map(fn => ({
      name: fn.name,
      line: lineOf(content, fn.start),
      cyclomatic: 1 + decisionPoints(fn.body),
      loc: fn.body.split('\n').length
    }));
    files.push({ file: relative(root, f), loc: content.split('\n').length, functions });
  }
  return files;
}

function scoreTests(tests) {
  let score = 100;
  score -= tests.failed * 20;
  score -= tests.slow * 15;
  return Math.max(0, score);
}

function scoreComplexity(functions) {
  if (functions.length === 0) return 100;
  const over = functions.filter(f => f.cyclomatic > CC_THRESHOLD).length;
  return Math.max(0, Math.round(100 * (1 - over / functions.length)));
}

const modules = discoverModules(root);
const moduleReports = [];
let allCases = [];
let allFiles = [];
let buildOk = true;

for (const mod of modules) {
  const result = mod.type === 'java' ? buildAndTestJava(mod) : buildAndTestNode(mod);
  const files = analyzeComplexity(mod);
  allCases = allCases.concat(result.cases.map(c => ({ ...c, module: relative(root, mod.dir) || '.' })));
  allFiles = allFiles.concat(files.map(f => ({ ...f, module: relative(root, mod.dir) || '.' })));
  if (!result.build.ok) buildOk = false;
  moduleReports.push({
    dir: relative(root, mod.dir) || '.',
    type: mod.type,
    build: { status: result.build.skipped ? 'skipped' : result.build.ok ? 'pass' : 'fail', ms: result.build.ms },
    tests: { ran: result.testRan, count: result.cases.length }
  });
}

const allFunctions = [];
for (const f of allFiles) {
  for (const fn of f.functions) {
    allFunctions.push({ module: f.module, file: f.file, name: fn.name, line: fn.line, cyclomatic: fn.cyclomatic, loc: fn.loc, exceeds: fn.cyclomatic > CC_THRESHOLD });
  }
}

const tests = {
  total: allCases.length,
  passed: allCases.filter(c => c.status === 'passed').length,
  failed: allCases.filter(c => c.status === 'failed').length,
  skipped: allCases.filter(c => c.status === 'skipped').length,
  slow: allCases.filter(c => c.slow).length,
  slowThresholdSeconds: SLOW_TEST_SECONDS,
  methods: allCases.sort((a, b) => b.durationMs - a.durationMs)
};

const sortedFns = allFunctions.sort((a, b) => b.cyclomatic - a.cyclomatic);
const complexity = {
  ccThreshold: CC_THRESHOLD,
  totalFunctions: allFunctions.length,
  overThreshold: allFunctions.filter(f => f.exceeds).length,
  maxCyclomatic: sortedFns.length ? sortedFns[0].cyclomatic : 0,
  avgCyclomatic: allFunctions.length ? Number((allFunctions.reduce((s, f) => s + f.cyclomatic, 0) / allFunctions.length).toFixed(2)) : 0,
  functions: sortedFns
};

const longFunctions = allFunctions.filter(f => f.loc > FUNC_LEN_THRESHOLD);
const longFiles = allFiles.filter(f => f.loc > FILE_LOC_THRESHOLD).map(f => ({ file: f.file, loc: f.loc }));

const deterministicRules = [
  { id: 'build-passes', category: 'build', status: buildOk ? 'pass' : 'fail', detail: buildOk ? 'all modules build' : 'one or more modules failed to build' },
  { id: 'tests-pass', category: 'tests', status: tests.failed === 0 ? 'pass' : 'fail', detail: tests.failed + ' failing tests' },
  { id: 'no-slow-tests', category: 'tests', status: tests.slow === 0 ? 'pass' : 'fail', detail: tests.slow + ' tests at or above ' + SLOW_TEST_SECONDS + 's', findings: tests.methods.filter(m => m.slow).map(m => ({ test: (m.classname ? m.classname + '#' : '') + m.name, durationMs: m.durationMs })) },
  { id: 'cyclomatic-complexity', category: 'complexity', status: complexity.overThreshold === 0 ? 'pass' : 'fail', detail: complexity.overThreshold + ' functions over CC ' + CC_THRESHOLD, findings: sortedFns.filter(f => f.exceeds).map(f => ({ where: f.file + ':' + f.line + ' ' + f.name, cyclomatic: f.cyclomatic })) },
  { id: 'function-length', category: 'codeQuality', status: longFunctions.length === 0 ? 'pass' : 'fail', detail: longFunctions.length + ' functions over ' + FUNC_LEN_THRESHOLD + ' lines', findings: longFunctions.map(f => ({ where: f.file + ':' + f.line + ' ' + f.name, loc: f.loc })) },
  { id: 'file-length', category: 'codeQuality', status: longFiles.length === 0 ? 'pass' : 'fail', detail: longFiles.length + ' files over ' + FILE_LOC_THRESHOLD + ' lines', findings: longFiles }
];

const scores = {
  build: buildOk ? 100 : 0,
  tests: scoreTests(tests),
  complexity: scoreComplexity(allFunctions)
};

const report = {
  meta: {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    target: root,
    slowThresholdSeconds: SLOW_TEST_SECONDS,
    ccThreshold: CC_THRESHOLD
  },
  languages: [...new Set(modules.map(m => m.type))],
  modules: moduleReports,
  metrics: {
    totalFiles: allFiles.length,
    totalLoc: allFiles.reduce((s, f) => s + f.loc, 0),
    totalFunctions: allFunctions.length
  },
  build: { status: buildOk ? 'pass' : 'fail' },
  tests,
  complexity,
  deterministicRules,
  scores
};

const outDir = join(root, '.lint');
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'deterministic.json'), JSON.stringify(report, null, 2));

console.log('target: ' + root);
console.log('languages: ' + report.languages.join(', '));
console.log('modules: ' + moduleReports.length + '  files: ' + report.metrics.totalFiles + '  functions: ' + report.metrics.totalFunctions);
console.log('build: ' + report.build.status);
console.log('tests: ' + tests.passed + '/' + tests.total + ' passed, ' + tests.failed + ' failed, ' + tests.slow + ' slow');
console.log('complexity: max ' + complexity.maxCyclomatic + ', avg ' + complexity.avgCyclomatic + ', ' + complexity.overThreshold + ' over threshold ' + CC_THRESHOLD);
console.log('scores: build ' + scores.build + ', tests ' + scores.tests + ', complexity ' + scores.complexity);
console.log('wrote ' + join(outDir, 'deterministic.json'));
