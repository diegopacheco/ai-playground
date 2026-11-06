#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[ -d "$ROOT/backend/.venv" ] || fail "backend is not set up, run ./scripts/setup.sh first"
[ -d "$ROOT/frontend/node_modules" ] || fail "frontend is not set up, run ./scripts/setup.sh first"

backend_port="$(service_port backend)"
ui_port="$(service_port ui)"

start_bg backend "$ROOT/backend" env POPPY_BASE_URL="$(service_url backend)" uv run granian --interface asgi --host 127.0.0.1 --port "$backend_port" poppy_company.app:app
wait_port_up "$backend_port" 60 || fail "backend did not open port $backend_port, see $LOGS/backend.log"

start_bg ui "$ROOT/frontend" bun run dev
wait_port_up "$ui_port" 60 || fail "ui did not open port $ui_port, see $LOGS/ui.log"

"$SCRIPTS/status.sh"

log "links"
printf "%-10s %s\n" ui "$(service_url ui /)"
printf "%-10s %s\n" discovery "$(service_url backend /.well-known/poppy.json)"
printf "%-10s %s\n" health "$(service_url backend /health)"
