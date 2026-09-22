#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"
require "$PYTHON"

( cd "$ROOT/agent-sdk" && PYTHONPATH=. "$PYTHON" -m unittest discover -s tests ) || fail "agent-sdk tests failed"
( cd "$ROOT/bridge" && "$PYTHON" -m unittest discover -s tests ) || fail "bridge tests failed"
( cd "$ROOT/skills" && "$PYTHON" -m unittest discover -s tests ) || fail "skills tests failed"
( cd "$ROOT/app" && "$PYTHON" -m unittest discover -s tests ) || fail "app tests failed"

started=0
if ! "$SCRIPTS/status.sh" >/dev/null; then
  "$SCRIPTS/start-all.sh" >/dev/null
  started=1
fi

status=0
BIFROST_PORT="$(service_port bifrost)" APP_PORT="$(service_port app)" "$PYTHON" -m unittest -v "$ROOT/tests/integration_test.py" || status=$?

if [ "$started" -eq 1 ]; then
  "$SCRIPTS/stop-all.sh" >/dev/null
fi

[ "$status" -eq 0 ] || fail "integration tests failed"
log "tests passed"
