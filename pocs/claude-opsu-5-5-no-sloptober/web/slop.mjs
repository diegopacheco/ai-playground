const COMMENT = /^\s*(#|\/\/|\/\*|\*|--|<!--)/;

export function commentRatio(diff) {
  const lines = diff
    .split("\n")
    .filter((line) => line.startsWith("+") && !line.startsWith("+++") && line.slice(1).trim())
    .map((line) => line.slice(1));
  if (lines.length < 5) return 0;
  return lines.filter((line) => COMMENT.test(line)).length / lines.length;
}

function verdictFor(rules, score) {
  if (score >= rules.block_at) return "BLOCK";
  if (score >= rules.suspect_at) return "SUSPECT";
  return "PASS";
}

export function score(rules, text, diff = "") {
  const hits = [];
  for (const rule of rules.rules) {
    const count = (text.match(new RegExp(rule.pattern, "gi")) || []).length;
    if (count) {
      hits.push({ id: rule.id, label: rule.label, count, points: Math.min(count * rule.weight, rule.cap) });
    }
  }
  const ratio = commentRatio(diff);
  for (const rule of rules.diff_rules) {
    if (ratio >= rule.min_ratio) {
      hits.push({ id: rule.id, label: rule.label, count: Math.round(ratio * 100) / 100, points: rule.weight });
    }
  }
  const total = Math.min(hits.reduce((sum, hit) => sum + hit.points, 0), 100);
  return { score: total, verdict: verdictFor(rules, total), hits };
}
