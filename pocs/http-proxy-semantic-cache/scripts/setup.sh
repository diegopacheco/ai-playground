#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require podman-compose
require cargo
require bun
require claude
require ollama

podman-compose pull

models="$(ollama list 2>/dev/null)" || fail "ollama is not running, start it with: ollama serve"
if ! printf "%s\n" "$models" | grep -q '^nomic-embed-text'; then
  ollama pull nomic-embed-text
fi

( cd "$ROOT/backend" && cargo build --release ) || fail "backend build failed"
( cd "$ROOT/ui" && bun install ) || fail "ui install failed"

if ! grep -q '^\.run/$' "$ROOT/.gitignore" 2>/dev/null; then
  printf ".run/\n" >>"$ROOT/.gitignore"
fi

log "setup done"
