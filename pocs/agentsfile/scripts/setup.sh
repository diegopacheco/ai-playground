#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for tool in cargo go uv bun mvn python3; do require "$tool"; done

log "rust sdk and linter"
cargo build -q --manifest-path "$ROOT/sdks/rust/Cargo.toml"
cargo build -q --release --manifest-path "$ROOT/linter/Cargo.toml"
log "python sdk"
( cd "$ROOT/sdks/python" && uv sync -q )
log "go sdk"
( cd "$ROOT/sdks/go" && go build ./... )
log "ts sdk"
( cd "$ROOT/sdks/ts" && bun install --silent )
log "java sdk"
( cd "$ROOT/sdks/java" && with_java25 mvn -q -DskipTests package )
log "setup done, linter at linter/target/release/agentsfile-lint"
