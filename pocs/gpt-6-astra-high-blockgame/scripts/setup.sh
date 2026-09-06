#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
command -v node >/dev/null || fail 'Node.js 22 or newer is required'
command -v npm >/dev/null || fail 'npm is required'
npm ci --ignore-scripts
npx playwright install chromium
printf 'Brisa is ready. Run ./scripts/start-all.sh\n'
