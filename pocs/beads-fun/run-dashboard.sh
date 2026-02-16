#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== Playwright Test Dashboard ==="
echo ""
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
sleep 1
rm -f "$SCRIPT_DIR/backend/twitter.db"
cd "$SCRIPT_DIR/frontend" && npx playwright test 2>&1
echo ""
echo "=== Opening HTML Report ==="
npx playwright show-report
