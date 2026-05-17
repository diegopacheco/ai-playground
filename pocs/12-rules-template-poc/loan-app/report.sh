#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

bun run report/generate.ts

REPORT="$(pwd)/report/index.html"
echo "report: file://$REPORT"

if command -v open >/dev/null 2>&1; then
  open "$REPORT"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$REPORT"
fi
