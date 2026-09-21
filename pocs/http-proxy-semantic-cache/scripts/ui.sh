#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port ui)"
[ -n "$port" ] || fail "ui is not declared in scripts/ports.env"

url="$(service_url ui)"
port_up "$port" || fail "ui is not running on $port, run ./scripts/start-all.sh first"

log "opening $url"
open_url "$url"
