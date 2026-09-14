#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for tool in node npm npx curl rg lsof; do
  command -v "$tool" >/dev/null || fail "$tool is required"
done
npm ci
npx playwright install chromium
printf '%s\n' 'Setup complete'
