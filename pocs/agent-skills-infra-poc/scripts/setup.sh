#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require uv
require bun
require podman
require podman-compose
require python3.15

log "installing infra python dependencies"
(cd "$INFRA" && uv sync -q -p 3.15)

log "installing evals ui dependencies"
(cd "$UI_DIR" && bun install --silent)

log "generating OAuth clients and signing key"
"$INFRA/bin/agentinfra" bootstrap >/dev/null

log "building containers"
compose_env
podman-compose build >"$LOGS/compose-build.log" 2>&1 || fail "container build failed, see $LOGS/compose-build.log"

log "setup done"
