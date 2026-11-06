#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

rust_sdk() { cargo test -q --manifest-path "$ROOT/sdks/rust/Cargo.toml"; }
linter() { cargo test -q --manifest-path "$ROOT/linter/Cargo.toml"; }
factory_cli() { cargo test -q --manifest-path "$ROOT/cli/Cargo.toml"; }
lint_factories() { cargo run -q --manifest-path "$ROOT/linter/Cargo.toml" -- "$ROOT/Factoryfile" "$ROOT"/samples/*.factoryfile; }
render_factories() {
  for file in "$ROOT/Factoryfile" "$ROOT"/samples/*.factoryfile; do
    cargo run -q --manifest-path "$ROOT/cli/Cargo.toml" -- --render "$file" --no-color >/dev/null
  done
  log "rendered $(ls "$ROOT"/samples/*.factoryfile | wc -l | tr -d ' ') samples and the Factoryfile"
}
python_sdk() { ( cd "$ROOT/sdks/python" && uv run pytest -q && uv run ty check ); }
go_sdk() { ( cd "$ROOT/sdks/go" && go vet ./... && go test -count=1 ./... ); }
ts_sdk() { ( cd "$ROOT/sdks/ts" && bun test && bunx tsc --noEmit ); }
java_sdk() { with_java25 mvn -f "$ROOT/sdks/java/pom.xml" test | grep -E "Tests run: [0-9]+, Failures" | tail -1; }

for suite in rust_sdk linter factory_cli lint_factories render_factories python_sdk go_sdk ts_sdk java_sdk; do
  log "== $suite"
  "$suite" || fail "$suite failed"
done
log "all suites passed"
