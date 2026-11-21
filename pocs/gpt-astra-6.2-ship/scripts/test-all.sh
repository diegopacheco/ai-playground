#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
npx playwright test scripts/scene.spec.cjs --workers=1 --output="${TMPDIR:-/tmp}/pirate-ship-results"
