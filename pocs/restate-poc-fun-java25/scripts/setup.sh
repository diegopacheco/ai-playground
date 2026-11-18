#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require podman
require podman-compose
require mvn
require bun
require curl
[ -x "$JAVA_BIN" ] || fail "Java 25 not found at $JAVA_HOME, set JAVA_HOME_25"
agent_env
require "$AGENT_PROVIDER"
(cd "$ROOT/agent-service" && mvn -q package -DskipTests) || fail "agent-service build failed"
(cd "$ROOT/ui" && bun install) || fail "ui install failed"
compose pull >"$LOGS/podman-compose.log" 2>&1 || fail "podman-compose pull failed, see $LOGS/podman-compose.log"
log "setup done"
