#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require cargo
require bun
require claude
require_ollama_model
start_redis

( cd "$ROOT/backend" && cargo test ) || fail "backend tests failed"
( cd "$ROOT/ui" && bun run typecheck ) || fail "ui typecheck failed"
( cd "$ROOT/ui" && bun run test ) || fail "ui tests failed"

log "tests passed"
