#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
command -v node >/dev/null || fail 'Node.js is required'
npm ci
npx playwright install chromium
printf '%s\n' 'Setup complete'
