#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "vetting and building the Go console and agent"
(cd "$ROOT/backend" && gofmt -l . | (! grep .) && go vet ./... && go build -o "$BIN/console" ./cmd/console && go build -o "$BIN/agent" ./cmd/agent)
log "backend built"
