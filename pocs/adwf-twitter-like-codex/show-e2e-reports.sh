#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$ROOT_DIR/frontend/playwright-report" ]; then
  "$ROOT_DIR/test-e2e.sh"
fi
cd "$ROOT_DIR/frontend"
bunx playwright show-report
