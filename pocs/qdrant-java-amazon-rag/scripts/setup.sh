#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require mvn
require bun
require podman-compose
require ollama
use_java25

log "java used by maven"
mvn -v

log "pulling qdrant image"
podman-compose pull

log "pulling ollama models"
ollama pull nomic-embed-text
ollama pull llama3.2

log "building backend"
( cd "$ROOT/backend" && mvn -q -DskipTests package )

log "installing ui dependencies"
( cd "$ROOT/ui" && bun install )

log "setup done"
