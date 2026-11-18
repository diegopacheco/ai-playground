#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

API="$(service_url agent-api)"
agent_env
container_running || fail "trigger.dev is not running, run ./scripts/start-all.sh"
port_up "$(service_port agent-api)" || fail "agent-api is not running, run ./scripts/start-all.sh"
pid_running trigger-dev || fail "trigger dev is not running, run ./scripts/start-all.sh"

json_field() {
  python3 -c "import json,sys; v=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'v': v}))" "$1"
}

SENT="$(curl -sf -X POST "$API/runs" -H 'content-type: application/json' -d '{"goal":"Name three fun facts about octopuses"}')" || fail "could not trigger the workflow"
RUN_ID="$(printf "%s" "$SENT" | json_field "v['runId']")"
log "workflow $RUN_ID triggered"

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

stop_tree trigger-dev KILL
log "trigger dev worker killed with SIGKILL mid run"
start_trigger_dev

tries=420
while [ "$tries" -gt 0 ]; do
  VIEW="$(curl -sf "$API/runs/$RUN_ID" || printf '{}')"
  STATUS="$(printf "%s" "$VIEW" | json_field "v.get('status', '')" 2>/dev/null || true)"
  if [ "$STATUS" = "COMPLETED" ] && printf "%s" "$VIEW" | grep -q '"phase": "done"'; then break; fi
  case "$STATUS" in FAILED|CRASHED|SYSTEM_FAILURE|CANCELED|EXPIRED|TIMED_OUT) fail "workflow ended $STATUS after the crash: $VIEW" ;; esac
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the workflow did not complete within 420s after the crash: $VIEW"
log "trigger.dev retried the workflow and it completed"

PLANS="$(curl -sf -G "$TRIGGER_API_URL/api/v1/runs" -H "authorization: Bearer $TRIGGER_SECRET_KEY" \
  --data-urlencode "filter[tag]=wf_$RUN_ID" --data-urlencode "filter[taskIdentifier]=plan-goal" | json_field "len(v['data'])")" || fail "could not list the plan runs"
[ "$PLANS" = "1" ] || fail "the plan task ran $PLANS times, the idempotency key should keep it at 1"
log "the plan task ran once: its result was reused through the idempotency key, not called again"
