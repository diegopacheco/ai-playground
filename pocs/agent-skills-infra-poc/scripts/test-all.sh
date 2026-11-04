#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for name in $API_SERVICES; do
  port_up "$(service_port "$name")" || fail "$name is down, run scripts/start-all.sh first"
done

log "== infra type check"
(cd "$INFRA" && uv run -q ty check .) || fail "infra type check failed"

log "== infra unit and live tests"
(cd "$INFRA" && PYTHONPATH="$INFRA" .venv/bin/python -m unittest discover -s tests -t .) || fail "infra tests failed"

log "== mcp tests"
python3.15 -m unittest discover -s "$ROOT/mcp/tests" -t "$ROOT/mcp" || fail "mcp tests failed"

log "== skills tests"
python3.15 -m unittest discover -s "$ROOT/skills/tests" -t "$ROOT/skills" || fail "skills tests failed"

log "== ui type check and tests"
(cd "$UI_DIR" && bun run typecheck && bun test src) || fail "ui tests failed"

log "== eval suites"
for pair in decision-log:skills/decision-log answer-grader:skills/answer-grader bookmarks-mcp:mcp/bookmarks-mcp feature-flags-mcp:mcp/feature-flags-mcp; do
  client="${pair%%:*}"
  dir="${pair#*:}"
  result="$("$INFRA/bin/agentinfra" --client "$client" eval run "$ROOT/$dir/evals/suite.json")" || fail "eval suite of $client could not run"
  score="$(printf "%s" "$result" | python3.15 -c 'import json,sys; d=json.load(sys.stdin); print(d["score"], d["passed"], d["failed"], d["id"])')"
  log "$client score passed failed run: $score"
  case "$score" in
    "1.0 "*) ;;
    *) fail "eval suite of $client is below 1.0" ;;
  esac
done

log "all tests passed"
