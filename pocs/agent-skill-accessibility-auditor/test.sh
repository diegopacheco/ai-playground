#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d skill/node_modules ]; then
  echo "installing skill dependencies"
  npm --prefix skill install --no-audit --no-fund
fi

if [ ! -d sample/node_modules ]; then
  echo "installing sample dependencies"
  npm --prefix sample install --no-audit --no-fund
fi

echo "starting sample app"
(npm --prefix sample run dev >/tmp/pixel-pantry.log 2>&1 &)

UP=""
for i in $(seq 1 60); do
  if curl -sf http://localhost:5188 >/dev/null; then UP="yes"; break; fi
  sleep 1
done

if [ -z "$UP" ]; then
  echo "sample app did not start"
  cat /tmp/pixel-pantry.log
  exit 1
fi

echo "running audit"
node skill/scripts/audit.mjs http://localhost:5188 "$ROOT/a11y-report"

PID="$(lsof -ti tcp:5188 || true)"
if [ -n "$PID" ]; then kill "$PID"; fi

test -f "$ROOT/a11y-report/index.html"
test -f "$ROOT/a11y-report/data.json"
test -f "$ROOT/a11y-report/screenshot.png"
echo "PASS report written to a11y-report/"
