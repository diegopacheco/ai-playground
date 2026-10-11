#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

[ -x "$ROOT/backend/.venv/bin/python" ] || fail "backend is not installed, run ./scripts/setup.sh first"
require bun

( cd "$ROOT/backend" && .venv/bin/python -m mypy ) || fail "backend type check failed"
( cd "$ROOT/backend" && .venv/bin/python -m unittest discover -s tests -t . ) || fail "backend tests failed"
( cd "$ROOT/ui" && bun run typecheck ) || fail "ui type check failed"
( cd "$ROOT/ui" && bun test src ) || fail "ui tests failed"

log "tests passed"
