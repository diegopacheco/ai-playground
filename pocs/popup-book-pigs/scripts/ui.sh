#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port frontend)"
[ -n "$port" ] || fail "frontend is not declared in scripts/ports.env"

url="http://localhost:$port"
port_up "$port" || fail "frontend is not running on $port, run ./scripts/start-all.sh first"

log "opening $url"
open_url "$url"
