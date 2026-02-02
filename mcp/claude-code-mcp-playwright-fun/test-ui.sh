#!/bin/bash
cd "$(dirname "$0")"

if [ ! -d "node_modules/@playwright" ]; then
    echo "Installing Playwright..."
    npm install
    npx playwright install chromium
fi

echo "Running UI tests..."
npx playwright test --reporter=list

echo ""
echo "Test complete. HTML report available at: playwright-report/index.html"
