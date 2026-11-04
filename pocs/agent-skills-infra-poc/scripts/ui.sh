#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port_up "$(service_port ui)" || fail "ui is not running, run scripts/start-all.sh"
log "eval-viewer credentials: $INFRA/.secrets/credentials/eval-viewer.env"
open_url "$(service_url ui)"
