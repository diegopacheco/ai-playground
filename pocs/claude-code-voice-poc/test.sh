#!/bin/bash
cd app

echo "=== Unit Tests ==="
bun test src/

echo ""
echo "=== E2E Tests (Playwright) ==="
bunx playwright install chromium --with-deps 2>/dev/null
bunx playwright test
