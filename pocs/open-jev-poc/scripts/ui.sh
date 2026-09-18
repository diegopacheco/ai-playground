#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port app)"
[ -n "$port" ] || fail "app is not declared in scripts/ports.env"
port_up "$port" || fail "app is not running on $port, run ./scripts/start-all.sh first"

url="$(service_url app)"
log "opening $url"
open_url "$url"
