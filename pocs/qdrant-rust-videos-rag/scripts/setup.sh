#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require cargo
require bun
require podman-compose
require ollama
require_ollama

log "pulling qdrant image"
podman-compose pull

log "pulling ollama models"
ollama pull nomic-embed-text
ollama pull llama3.2

log "building backend"
( cd "$ROOT/backend" && cargo build --release )

log "installing ui dependencies"
( cd "$ROOT/ui" && bun install )

log "setup done"
