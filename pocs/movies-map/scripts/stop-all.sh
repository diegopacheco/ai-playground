#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_bg api
port_up "$(service_port api)" && fail "port $(service_port api) is still in use"
log "all services stopped"
