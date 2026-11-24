#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun
bunx tsc -p . || fail "typecheck failed"
bun test || fail "tests failed"
