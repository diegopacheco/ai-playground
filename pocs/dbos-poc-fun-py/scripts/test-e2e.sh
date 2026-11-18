#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

API="$(service_url agent-service)"
agent_env
container_running || fail "postgres is not running, run ./scripts/start-all.sh"
port_up "$(service_port agent-service)" || fail "agent-service is not running, run ./scripts/start-all.sh"

json_field() {
  python3 -c "import json,sys; v=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'v': v}))" "$1"
}

SENT="$(curl -sf -X POST "$API/runs" -H 'content-type: application/json' -d '{"goal":"Name three fun facts about octopuses"}')" || fail "could not start the workflow"
RUN_ID="$(printf "%s" "$SENT" | json_field "v['runId']")"
log "workflow $RUN_ID started"

tries=90
while [ "$tries" -gt 0 ]; do
  STATE="$(curl -sf "$API/runs/$RUN_ID" || true)"
  if printf "%s" "$STATE" | grep -q '"phase": "failed"'; then fail "workflow failed: $STATE"; fi
  if printf "%s" "$STATE" | grep -q '"status": "done"'; then break; fi
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the agent did not finish its first task within 90s: $STATE"
log "agent planned the goal and finished its first task"

CRASH_MS="$(python3 -c 'import time; print(time.time_ns() // 1_000_000)')"
kill_agent_service
log "agent-service killed with SIGKILL mid run"
start_agent_service
log "agent-service restarted"

tries=300
while [ "$tries" -gt 0 ]; do
  JOURNAL="$(curl -sf "$API/runs/$RUN_ID/journal" || printf '{}')"
  STATUS="$(printf "%s" "$JOURNAL" | json_field "v.get('status', '')" 2>/dev/null || true)"
  if [ "$STATUS" = "SUCCESS" ]; then break; fi
  if [ "$STATUS" = "ERROR" ]; then fail "workflow ended in ERROR after recovery: $JOURNAL"; fi
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the recovered workflow did not finish within 300s: $JOURNAL"
log "DBOS recovered the workflow and it finished"

PLAN_DONE="$(printf "%s" "$JOURNAL" | json_field "[s['completed_at'] for s in v['steps'] if s['name'] == 'plan'][0]")"
[ "$PLAN_DONE" -lt "$CRASH_MS" ] || fail "the plan step ran again after the crash ($PLAN_DONE >= $CRASH_MS)"
log "the plan step finished before the crash and was replayed from the journal, not called again"

STATE="$(curl -sf "$API/runs/$RUN_ID")"
printf "%s" "$STATE" | grep -q '"phase": "done"' || fail "final state is not done: $STATE"
log "final answer published"
