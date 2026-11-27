#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port_up "$(service_port console)" || fail "console is not running, run scripts/start-all.sh first"
log "e2e against the live kind cluster"
(cd "$ROOT/backend" && CONSOLE_URL="$(service_url console)" go test -tags e2e -count=1 -timeout 10m -v ./e2e/)
