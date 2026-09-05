#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
npm test
npx playwright test "$@"
