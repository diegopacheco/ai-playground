const path = require('path');

const results = require(path.resolve(process.argv[2])).results.results;
const check = process.argv[3];

const byPrompt = (label) => results.filter((r) => r.prompt.label === label);
const passed = (rows) => rows.filter((r) => r.success).length;
const metric = (rows, name) => rows.filter((r) => r.namedScores[name] === 1).length;

const report = [];
let ok = true;

function require_(condition, message) {
  report.push(`${condition ? 'PASS' : 'FAIL'}  ${message}`);
  if (!condition) ok = false;
}

if (check === 'classify') {
  const guided = byPrompt('guided');
  const terse = byPrompt('terse');
  const guidedRate = passed(guided) / guided.length;
  const terseRate = passed(terse) / terse.length;

  require_(
    guidedRate > terseRate,
    `the guided prompt beats the terse one (guided ${(guidedRate * 100).toFixed(0)}% vs terse ${(terseRate * 100).toFixed(0)}%)`,
  );
  require_(guidedRate >= 0.5, `the guided prompt triages at least half the tickets correctly (${(guidedRate * 100).toFixed(0)}%)`);
  require_(
    metric(guided, 'label-only') === guided.length,
    `the guided prompt always answers with a bare label (${metric(guided, 'label-only')}/${guided.length})`,
  );
} else if (check === 'extract') {
  require_(
    metric(results, 'valid-json') === results.length,
    `every reply is schema-valid JSON (${metric(results, 'valid-json')}/${results.length})`,
  );
  require_(
    metric(results, 'summary-length') === results.length,
    `every summary honours the word budget (${metric(results, 'summary-length')}/${results.length})`,
  );
} else {
  throw new Error(`unknown check: ${check}`);
}

console.log(report.join('\n'));
process.exit(ok ? 0 : 1);
