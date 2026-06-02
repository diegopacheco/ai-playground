#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const WEIGHTS = { build: 20, tests: 20, complexity: 15, principles: 15, bestPractices: 12, codeQuality: 10, naming: 8 };

const root = process.argv[2]
  ? (process.argv[2].startsWith('/') ? process.argv[2] : join(process.cwd(), process.argv[2]))
  : process.cwd();
const lintDir = join(root, '.lint');

const det = JSON.parse(readFileSync(join(lintDir, 'deterministic.json'), 'utf8'));
let sem = { rules: [], scores: {} };
try {
  sem = JSON.parse(readFileSync(join(lintDir, 'semantic.json'), 'utf8'));
} catch {
  sem = { rules: [], scores: {} };
}

const detRules = det.deterministicRules.map(r => ({
  id: r.id,
  category: r.category,
  type: 'deterministic',
  severity: r.status === 'pass' ? 'info' : 'warn',
  status: r.status,
  detail: r.detail,
  findings: r.findings || [],
  samples: r.samples || null
}));
const semRules = (sem.rules || []).map(r => ({ ...r, type: 'semantic' }));
const rules = [...detRules, ...semRules];

const scores = {
  build: det.scores.build,
  tests: det.scores.tests,
  complexity: det.scores.complexity,
  naming: sem.scores.naming ?? null,
  principles: sem.scores.principles ?? null,
  bestPractices: sem.scores.bestPractices ?? null,
  codeQuality: sem.scores.codeQuality ?? null
};

let weighted = 0;
let weightSum = 0;
for (const [category, weight] of Object.entries(WEIGHTS)) {
  if (scores[category] != null) {
    weighted += scores[category] * weight;
    weightSum += weight;
  }
}
const overall = weightSum ? Math.round(weighted / weightSum) : 0;

const report = {
  meta: det.meta,
  languages: det.languages,
  modules: det.modules,
  metrics: det.metrics,
  build: det.build,
  tests: det.tests,
  complexity: det.complexity,
  rules,
  scores: { overall, byCategory: scores }
};

writeFileSync(join(lintDir, 'report.json'), JSON.stringify(report, null, 2));

const historyDir = join(lintDir, 'history');
mkdirSync(historyDir, { recursive: true });
const stamp = det.meta.generatedAt.replace(/[:.]/g, '-');
const historyEntry = {
  generatedAt: det.meta.generatedAt,
  overall,
  byCategory: scores,
  tests: { total: det.tests.total, passed: det.tests.passed, failed: det.tests.failed, slow: det.tests.slow },
  complexity: { maxCyclomatic: det.complexity.maxCyclomatic, avgCyclomatic: det.complexity.avgCyclomatic, overThreshold: det.complexity.overThreshold }
};
writeFileSync(join(historyDir, stamp + '.json'), JSON.stringify(historyEntry, null, 2));

console.log('overall score: ' + overall);
console.log('by category: ' + JSON.stringify(scores));
console.log('rules: ' + rules.length + ' total, ' + rules.filter(r => r.status === 'pass').length + ' pass, ' + rules.filter(r => r.status !== 'pass').length + ' flagged');
console.log('wrote ' + join(lintDir, 'report.json') + ' and history entry ' + stamp + '.json');
