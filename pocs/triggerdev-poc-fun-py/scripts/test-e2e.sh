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

descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null || true); do
    printf "%s\n" "$child"
    descendants "$child"
  done
}

agent_cli_pid() {
  local pid
  for pid in $(descendants "$(cat "$RUN/trigger-dev.pid")"); do
    if [ "$(basename "$(ps -o comm= -p "$pid" 2>/dev/null || true)")" = "$AGENT_PROVIDER" ]; then
      printf "%s\n" "$pid"
      return 0
    fi
  done
  return 1
}

tries=60
while [ "$tries" -gt 0 ]; do
  CLI_PID="$(agent_cli_pid || true)"
  [ -n "$CLI_PID" ] && break
  sleep 1
  tries=$((tries - 1))
done
[ -n "$CLI_PID" ] || fail "no $AGENT_PROVIDER process found under the trigger dev worker"
kill -KILL "$CLI_PID"
log "killed the $AGENT_PROVIDER CLI (pid $CLI_PID) in the middle of task 2"

tries=420
while [ "$tries" -gt 0 ]; do
  VIEW="$(curl -sf "$API/runs/$RUN_ID" || printf '{}')"
  STATUS="$(printf "%s" "$VIEW" | json_field "v.get('status', '')" 2>/dev/null || true)"
  if [ "$STATUS" = "COMPLETED" ] && printf "%s" "$VIEW" | grep -q '"phase": "done"'; then break; fi
  case "$STATUS" in FAILED|CRASHED|SYSTEM_FAILURE|CANCELED|EXPIRED|TIMED_OUT) fail "workflow ended $STATUS after the killed step: $VIEW" ;; esac
  sleep 1
  tries=$((tries - 1))
done
[ "$tries" -gt 0 ] || fail "the workflow did not complete within 420s after the killed step: $VIEW"
log "the workflow completed"

runs_of() {
  curl -sf -G "$TRIGGER_API_URL/api/v1/runs" -H "authorization: Bearer $TRIGGER_SECRET_KEY" \
    --data-urlencode "filter[tag]=wf_$RUN_ID" --data-urlencode "filter[taskIdentifier]=$1" --data-urlencode "page[size]=50"
}

PLANS="$(runs_of plan-goal | json_field "len(v['data'])")" || fail "could not list the plan runs"
[ "$PLANS" = "1" ] || fail "the plan task ran $PLANS times, the idempotency key should keep it at 1"
log "the plan task ran once: no step before the failure was called again"

ATTEMPTS="$(runs_of execute-task | json_field "' '.join(r['id'] for r in v['data'])" | tr ' ' '\n' | while read -r child; do
  curl -sf "$TRIGGER_API_URL/api/v3/runs/$child" -H "authorization: Bearer $TRIGGER_SECRET_KEY" | json_field "v['attemptCount']"
done | sort -n | tail -1)"
[ "${ATTEMPTS:-0}" -ge 2 ] || fail "no execute-task run was retried, max attempts $ATTEMPTS"
log "the killed task was retried by trigger.dev (attempt $ATTEMPTS) and finished"
