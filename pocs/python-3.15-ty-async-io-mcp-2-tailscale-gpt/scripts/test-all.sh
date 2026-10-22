#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

uv run ty check src tests || fail "ty type check failed"
uv run pytest -v || fail "pytest failed"

if port_up "$(service_port mcp)"; then
  expect() {
    local expected actual
    expected="$1"
    shift
    actual="$("$SCRIPTS/calc.sh" "$@")" || fail "live call failed: $*"
    [ "$actual" = "$expected" ] || fail "live call $* returned $actual, expected $expected"
    log "live $* = $actual"
  }
  expect 3 add 1 2
  expect 9 evaluate "(1 + 2) * 3"
  expect 9 polish "* + 1 2 3"
  expect 3 english "one plus two"
else
  log "mcp is down, live HTTP checks skipped"
fi
log "all tests passed"
