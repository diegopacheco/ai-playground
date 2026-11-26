#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "starting"

ensure_ollama
log "ollama up on $(service_url ollama)"

backend_port="$(service_port backend)"
frontend_port="$(service_port frontend)"

if ! port_up "$backend_port"; then
  ( cd "$ROOT/backend" && mvn -q -DskipTests package ) || fail "backend build failed"
fi
start_bg backend "$ROOT/backend" env BACKEND_PORT="$backend_port" OLLAMA_URL="$(service_url ollama)" java -jar target/decisions-service-1.0.0.jar
wait_port_up "$backend_port" 60 || fail "backend did not open port $backend_port, see $LOGS/backend.log"
log "backend up on $(service_url backend)"

start_bg frontend "$ROOT/frontend" env FRONTEND_PORT="$frontend_port" BACKEND_PORT="$backend_port" bun run dev
wait_port_up "$frontend_port" 60 || fail "frontend did not open port $frontend_port, see $LOGS/frontend.log"
log "frontend up on $(service_url frontend)"

"$SCRIPTS/status.sh"

log "links"
for name in $(service_names); do
  printf "%-14s %s\n" "$name" "$(service_url "$name")"
done
