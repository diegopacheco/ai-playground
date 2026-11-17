#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

URL="$(service_url ui)"
port_up "$(service_port ui)" || fail "ui is DOWN, run scripts/start-all.sh first"
open_url "$URL"
log "opened $URL"
