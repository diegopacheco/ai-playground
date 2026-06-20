#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "node is required"
  exit 1
fi

echo "running analyzer against $HOME/.claude"
node skill/scripts/analyze.mjs habit-report

test -f "$ROOT/habit-report/index.html"
test -f "$ROOT/habit-report/data.json"

node -e '
const d = require("./habit-report/data.json");
if (!d.totals || d.totals.events <= 0) { console.error("no interactions counted"); process.exit(1); }
if (!d.insights || d.insights.length < 5) { console.error("expected at least 5 insights"); process.exit(1); }
if (!d.days || Object.keys(d.days).length === 0) { console.error("contribution grid is empty"); process.exit(1); }
console.log("checks passed: " + d.totals.events + " interactions, " + d.insights.length + " insights, " + Object.keys(d.days).length + " active days");
'

echo "PASS report written to habit-report/"
