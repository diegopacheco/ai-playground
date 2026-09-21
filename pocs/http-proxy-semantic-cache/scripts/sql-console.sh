#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port="$(service_port redis)"
[ -n "$port" ] || fail "redis is not declared in scripts/ports.env"
port_up "$port" || fail "redis is not running on $port, run ./scripts/start-all.sh first"

if command -v redis-cli >/dev/null 2>&1; then
  redis-cli -h localhost -p "$port" "$@"
else
  podman exec -it semantic-cache-redis redis-cli "$@"
fi
