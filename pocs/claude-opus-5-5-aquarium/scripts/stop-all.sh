#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_bg web
port_up "$(service_port web)" && fail "web still listening on $(service_port web)"
log "stopped"
