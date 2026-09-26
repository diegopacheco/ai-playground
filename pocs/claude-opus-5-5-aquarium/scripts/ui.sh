#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port web)"
port_up "$port" || fail "web is not running on $port, run ./scripts/start-all.sh first"
log "opening $(service_url web)"
open_url "$(service_url web)"
