#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require uv
require bun
require podman-compose
require ollama
require_ollama

podman-compose pull
ollama pull nomic-embed-text
ollama pull llama3.2

( cd "$ROOT/backend" && uv sync --python 3.14 )
( cd "$ROOT/ui" && bun install )

log "setup done"
