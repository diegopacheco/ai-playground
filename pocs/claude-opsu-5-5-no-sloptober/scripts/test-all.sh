#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require python3
require node
log "skill tests"
python3 -m unittest discover -s "$ROOT/.claude/skills/no-slop-pr/tests" -v || fail "skill tests failed"
log "page tests"
node --test "$ROOT"/tests/*.test.mjs || fail "page tests failed"
log "tests passed"
