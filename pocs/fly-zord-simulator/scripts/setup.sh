#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun

log "installing workspace dependencies"
bun install

log "building the engine with vite"
bun run --filter '@fly-zord/engine' build

log "building the agents package with typescript"
bun run --filter '@fly-zord/agents' build

log "setup done"
