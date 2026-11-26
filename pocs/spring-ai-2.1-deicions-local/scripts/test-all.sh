#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

ensure_ollama

( cd "$ROOT/backend" && mvn test ) || fail "backend tests failed"
( cd "$ROOT/frontend" && bun run build && bun run typecheck && bun run test ) || fail "frontend tests failed"

log "tests passed"
