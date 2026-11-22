#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
npm ci
PLAYWRIGHT_SKIP_BROWSER_GC=1 npx playwright install chromium
