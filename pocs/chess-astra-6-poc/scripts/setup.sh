#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require bun
require npx
bun install --frozen-lockfile
npx playwright install chromium
printf 'Setup complete.\n'
