#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

INGRESS="$(service_url restate-ingress)"
ADMIN="$(service_url restate-admin)"
container_running || fail "restate is not running, run ./scripts/start-all.sh"
port_up "$(service_port agent-service)" || fail "agent-service is not running, run ./scripts/start-all.sh"

RUN_ID="e2e-$(date +%s)"
SENT="$(curl -sf -X POST "$INGRESS/AgentWorkflow/$RUN_ID/run/send" -H 'content-type: application/json' -d '{"goal":"Name three fun facts about octopuses"}')" || fail "could not start the workflow"
printf "%s" "$SENT" | grep -q '"invocationId"' || fail "restate did not return an invocation id: $SENT"
log "workflow $RUN_ID accepted"

tries=60
while [ "$tries" -gt 0 ]; do
  STATUS="$(curl -sf -X POST "$INGRESS/AgentWorkflow/$RUN_ID/status" || true)"
  if printf "%s" "$STATUS" | grep -q '"phase":"failed"'; then fail "workflow failed: $STATUS"; fi
  if printf "%s" "$STATUS" | grep -q '"status":"done"'; then break; fi
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the agent did not finish its first task within 60s: $STATUS"
log "agent planned the goal and finished its first task"

SQL="$(curl -sf -X POST "$ADMIN/query" -H 'content-type: application/json' -H 'accept: application/json' \
  -d "{\"query\":\"SELECT id, status FROM sys_invocation WHERE target_service_key = '$RUN_ID' AND target_handler_name = 'run'\"}")" || fail "admin query failed"
printf "%s" "$SQL" | grep -q 'inv_' || fail "restate has no invocation for $RUN_ID: $SQL"
log "restate tracks the invocation: $SQL"
