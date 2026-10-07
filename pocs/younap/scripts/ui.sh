#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port younap)"
[ -n "$port" ] || fail "younap is not declared in scripts/ports.env"

url="$(service_url younap)"
port_up "$port" || fail "younap is not running on $port, run ./scripts/start-all.sh first"

log "opening $url"
open_url "$url"
