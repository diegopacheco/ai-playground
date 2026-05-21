#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
LOG="$HERE/test.log"
t0=$(bp_now_sec)
RC=0
eval "$TEST_CMD" >"$LOG" 2>&1 || RC=$?
t1=$(bp_now_sec)
dur=$((t1 - t0))
SUMMARY=$(grep -E '[0-9]+ passed|[0-9]+ failed' "$LOG" | tail -n 1 || true)
PASSED=$( { echo "$SUMMARY" | grep -oE '[0-9]+ passed' || true; } | grep -oE '[0-9]+' | head -1 || true)
FAILED=$( { echo "$SUMMARY" | grep -oE '[0-9]+ failed' || true; } | grep -oE '[0-9]+' | head -1 || true)
SKIPPED=$( { echo "$SUMMARY" | grep -oE '[0-9]+ skipped' || true; } | grep -oE '[0-9]+' | head -1 || true)
PASSED=${PASSED:-0}
FAILED=${FAILED:-0}
SKIPPED=${SKIPPED:-0}
TOTAL=$((PASSED + FAILED + SKIPPED))
PASS=false
[ "$RC" -eq 0 ] && [ "$FAILED" -eq 0 ] && [ "$TOTAL" -gt 0 ] && PASS=true
{
  echo "TESTS_PASS=$PASS"
  echo "TESTS_TOTAL=$TOTAL"
  echo "TESTS_PASSED=$PASSED"
  echo "TESTS_FAILED=$FAILED"
  echo "TESTS_SKIPPED=$SKIPPED"
  echo "TESTS_DURATION=$dur"
} > .tests.env
if [ "$RC" -ne 0 ]; then
  bp_log_tail "$LOG" 60
fi
exit "$RC"
