#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "stopping"
stop_bg frontend
stop_bg console
pkill -f "$BIN/kubectl-ate --context $KCTX proxy actor" 2>/dev/null || true
for name in console frontend; do
  port="$(service_port "$name")"
  if port_up "$port"; then
    fail "$name still listening on $port"
  fi
done
log "stopped, the kind cluster and registry keep running"
