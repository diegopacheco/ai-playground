#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
npm test
npx playwright test
printf 'All configured tests passed. Use npm run test:live to include a real Codex call.\n'
