#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require java
require mvn
require bun
require ollama

ensure_port="$(service_port ollama)"
if ! port_up "$ensure_port" && command -v open >/dev/null 2>&1; then
  open -g -a Ollama || fail "could not launch Ollama"
  wait_port_up "$ensure_port" 60 || fail "ollama did not open port $ensure_port"
fi
ollama list | grep "^$DECISION_MODEL" >/dev/null || ollama pull "$DECISION_MODEL"

( cd "$ROOT/backend" && mvn -q -DskipTests package ) || fail "backend build failed"
( cd "$ROOT/frontend" && bun install ) || fail "frontend install failed"

if ! grep -q '^\.run/$' "$ROOT/.gitignore" 2>/dev/null; then
  printf ".run/\n" >>"$ROOT/.gitignore"
fi

log "setup done"
