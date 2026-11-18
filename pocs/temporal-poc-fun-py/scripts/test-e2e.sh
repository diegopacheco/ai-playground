#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

API="$(service_url agent-api)"
agent_env
container_running || fail "temporal is not running, run ./scripts/start-all.sh"
port_up "$(service_port agent-api)" || fail "agent-api is not running, run ./scripts/start-all.sh"
pid_running worker || fail "the worker is not running, run ./scripts/start-all.sh"

json_field() {
  python3 -c "import json,sys; v=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'v': v}))" "$1"
}

SENT="$(curl -sf -X POST "$API/runs" -H 'content-type: application/json' -d '{"goal":"Name three fun facts about octopuses"}')" || fail "could not start the workflow"
RUN_ID="$(printf "%s" "$SENT" | json_field "v['runId']")"
log "workflow $RUN_ID started"

tries=120
while [ "$tries" -gt 0 ]; do
  VIEW="$(curl -sf "$API/runs/$RUN_ID" || true)"
  if printf "%s" "$VIEW" | grep -q '"phase": "failed"'; then fail "workflow failed: $VIEW"; fi
  if printf "%s" "$VIEW" | grep -q '"status": "done"'; then break; fi
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the agent did not finish its first task within 120s: $VIEW"
log "agent planned the goal and finished its first task"

stop_tree worker KILL
log "worker killed with SIGKILL mid run"
start_worker
log "worker restarted"

tries=420
while [ "$tries" -gt 0 ]; do
  VIEW="$(curl -sf "$API/runs/$RUN_ID" || printf '{}')"
  STATUS="$(printf "%s" "$VIEW" | json_field "v.get('status', '')" 2>/dev/null || true)"
  if [ "$STATUS" = "COMPLETED" ]; then break; fi
  case "$STATUS" in FAILED|TERMINATED|CANCELED|TIMED_OUT) fail "workflow ended $STATUS after the crash: $VIEW" ;; esac
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the workflow did not complete within 420s after the crash: $VIEW"
printf "%s" "$VIEW" | grep -q '"phase": "done"' || fail "final state is not done: $VIEW"
log "temporal replayed the workflow on the new worker and it completed"

HISTORY="$(podman exec "$TEMPORAL_CONTAINER" temporal workflow show --workflow-id "$RUN_ID" --output json)" || fail "could not read the workflow history"
SUMMARY="$(printf "%s" "$HISTORY" | python3 -c '
import json, sys
events = json.load(sys.stdin)["events"]
plans = [e for e in events if e.get("activityTaskScheduledEventAttributes", {}).get("activityType", {}).get("name") == "plan_goal"]
attempts = [e["activityTaskStartedEventAttributes"].get("attempt", 1) for e in events if "activityTaskStartedEventAttributes" in e]
print(len(plans), max(attempts))
')"
PLANS="${SUMMARY% *}"
MAX_ATTEMPT="${SUMMARY#* }"
[ "$PLANS" = "1" ] || fail "plan_goal was scheduled $PLANS times, the history replay should keep it at 1"
log "plan_goal was scheduled once: the new worker replayed it from the event history"
[ "$MAX_ATTEMPT" -ge 2 ] || fail "no activity needed a second attempt, the kill did not hit a running activity"
log "the activity cut by the crash timed out on its heartbeat and finished on attempt $MAX_ATTEMPT"
