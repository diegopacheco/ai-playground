#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$SCRIPTS/unit-test.sh"
"$SCRIPTS/integration-test.sh"
if port_up "$(service_port frontend)"; then
  "$SCRIPTS/e2e-test.sh"
else
  fail "e2e tests need the app running, run scripts/start-all.sh first"
fi
log "all tests passed"
