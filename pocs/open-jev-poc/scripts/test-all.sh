#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[ -x "$PY" ] || fail "run ./scripts/setup.sh first"

log "tests started"
"$PY" -m pytest -q "$ROOT/tests/test_decide.py" "$ROOT/tests/test_server.py" || fail "unit tests failed"
"$PY" -m pytest -q "$ROOT/tests/test_model.py" || fail "model tests failed"
log "tests passed"
