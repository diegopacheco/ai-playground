#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port game)"
[ -n "$port" ] || fail "game is not declared in scripts/ports.env"

url="$(service_url game)"
port_up "$port" || fail "game is not running on $port, run ./scripts/start-all.sh first"

log "opening $url"
open_url "$url"
