#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

url="$(service_url arena)"
if ! port_up "$(service_port arena)"; then
  log "arena is not running, start it with ./scripts/start-all.sh"
fi
open_url "$url"
