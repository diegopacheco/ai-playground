#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/e2e-tests"
if [ ! -d "node_modules" ]; then
  bun install 2>&1
fi
npx playwright install chromium 2>&1
npx playwright test 2>&1
echo "E2E TESTS PASSED"
