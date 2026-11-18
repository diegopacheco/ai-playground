#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

agent_env
container_running || fail "trigger.dev is not running, run ./scripts/start-all.sh"
[ -n "${TRIGGER_USER_EMAIL:-}" ] || fail "no local user yet, run ./scripts/start-all.sh"
SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
curl -sf -o /dev/null -X POST "$TRIGGER_API_URL/login/magic" --data-urlencode "email=$TRIGGER_USER_EMAIL" --data-urlencode "action=send" \
  || fail "could not request a magic link"
LINK=""
tries=15
while [ -z "$LINK" ] && [ "$tries" -gt 0 ]; do
  LINK="$(podman logs --since "$SINCE" "$WEBAPP_CONTAINER" 2>&1 | grep -o "$TRIGGER_API_URL/magic?token=[^ ]*" | tail -1 || true)"
  sleep 1
  tries=$((tries - 1))
done
[ -n "$LINK" ] || fail "no magic link in the webapp logs"
log "logging $TRIGGER_USER_EMAIL into the local trigger.dev dashboard"
open_url "$LINK"
