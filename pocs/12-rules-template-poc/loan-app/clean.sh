#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

rm -rf backend/target
rm -rf frontend/test-results
rm -rf e2e/test-results
rm -rf e2e/playwright-report
rm -rf .run
rm -f report/index.html

echo "cleaned: surefire reports, jest results, playwright results/report, k6 summary, .run logs, aggregate report"
