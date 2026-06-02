export function scoreColor(score) {
  if (score == null) return '#94a3b8';
  if (score >= 85) return '#16a34a';
  if (score >= 70) return '#f59e0b';
  return '#dc2626';
}

export function statusColor(status) {
  if (status === 'pass') return '#16a34a';
  if (status === 'fail') return '#dc2626';
  return '#f59e0b';
}

export const CATEGORY_LABELS = {
  build: 'Build',
  tests: 'Tests',
  complexity: 'Complexity',
  naming: 'Naming',
  principles: 'Principles',
  bestPractices: 'Best Practices',
  codeQuality: 'Code Quality'
};

export function formatMs(ms) {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + 's';
  return ms + 'ms';
}
