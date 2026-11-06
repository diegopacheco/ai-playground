#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

rust_sdk() { cargo test -q --manifest-path "$ROOT/sdks/rust/Cargo.toml"; }
linter() { cargo test -q --manifest-path "$ROOT/linter/Cargo.toml"; }
lint_agentsfile() { cargo run -q --manifest-path "$ROOT/linter/Cargo.toml" -- "$ROOT/Agentsfile"; }
python_sdk() { ( cd "$ROOT/sdks/python" && uv run pytest -q && uv run ty check ); }
go_sdk() { ( cd "$ROOT/sdks/go" && go vet ./... && go test -count=1 ./... ); }
ts_sdk() { ( cd "$ROOT/sdks/ts" && bun test && bunx tsc --noEmit ); }
java_sdk() { with_java25 mvn -q -f "$ROOT/sdks/java/pom.xml" test; }

for suite in rust_sdk linter lint_agentsfile python_sdk go_sdk ts_sdk java_sdk; do
  log "== $suite"
  "$suite" || fail "$suite failed"
done
log "all suites passed"
