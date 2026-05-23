#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v bun >/dev/null || { echo "ERROR: bun not installed"; exit 1; }

echo "runner: typecheck"
(cd runner && bun run typecheck)

echo "runner: tests"
(cd runner && bun test)

if [ -d web ]; then
  echo "web: typecheck"
  (cd web && bun run typecheck)
  echo "web: tests"
  (cd web && bun test)
fi

echo "all tests passed"
