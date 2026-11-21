#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "tests started"

require go
require bun

( cd "$ROOT/backend" && gofmt -l . | tee "$RUN/gofmt.out" && [ ! -s "$RUN/gofmt.out" ] ) || fail "gofmt found unformatted files"
( cd "$ROOT/backend" && go vet ./... ) || fail "go vet failed"
( cd "$ROOT/backend" && go test -count=1 ./... ) || fail "backend tests failed"
( cd "$ROOT/frontend" && bun run typecheck ) || fail "frontend typecheck failed"
( cd "$ROOT/frontend" && bun test src ) || fail "frontend tests failed"

log "tests passed"
