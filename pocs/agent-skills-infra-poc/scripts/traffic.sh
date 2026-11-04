#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require python3.15
for name in $API_SERVICES; do
  port_up "$(service_port "$name")" || fail "$name is down, run scripts/start-all.sh"
done

rounds="${1:-0}"
case "$rounds" in
  ''|*[!0-9]*) fail "usage: scripts/traffic.sh [rounds], 0 or nothing runs until Ctrl+C" ;;
esac

if [ "$rounds" -eq 0 ]; then
  log "generating traffic through bookmarks-mcp, feature-flags-mcp, decision-log and answer-grader until Ctrl+C"
fi
exec python3.15 "$SCRIPTS/traffic.py" --rounds "$rounds"
